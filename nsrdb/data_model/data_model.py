"""The data model handles NSRDB data sources, with the end goal of preparing
data inputs to the NSRDB all-sky irradiance models.

A primary function of this model is to perform spatial and temporal
interpolation of ancillary data. MERRA2 and Asymmetry data are both spatially
and temporally interpolated to the NSRDB reference grid. Cloud properties
originating from the GOES satellites and processed by the University of
Wisconsin are mapped to the NSRDB grid via nearest neighbors (ReGrid).

Non-standard module dependencies (pip-installable):
    - netCDF4

The following variables can be processed using this module. For details see the
class variables in Ancillary() below.
    ('alpha',
     'asymmetry',
     'aod',
     'cloud_type',
     'cld_opd_dcomp',
     'cld_reff_dcomp',
     'cld_press_acha',
     'surface_pressure',
     'relative_humidity',
     'ssa',
     'ozone',
     'total_precipitable_water',
     'solar_zenith_angle',
     'air_temperature',
     'specific_humidity',
     'wind_speed',
     'wind_direction',
     'wind_speed_10m',
     'wind_direction_10m',
     'dew_point')
"""

import copy
import logging
import os
import time
from concurrent.futures import as_completed
from glob import glob
from typing import ClassVar
from warnings import warn

import numpy as np
import pandas as pd
import psutil
from rex.utilities.execution import SpawnProcessPool

from nsrdb import DATADIR
from nsrdb.data_model.base_handler import BaseDerivedVar
from nsrdb.data_model.clouds import CloudVar
from nsrdb.data_model.variable_factory import VarFactory
from nsrdb.file_handlers.file_system import NSRDBFileSystem as NSRDBfs
from nsrdb.file_handlers.outputs import Outputs
from nsrdb.utilities.file_utils import clean_meta
from nsrdb.utilities.interpolation import (
    parse_method,
    spatial_interp,
    temporal_lin,
    temporal_step,
)
from nsrdb.utilities.nearest_neighbor import geo_nn, knn

logger = logging.getLogger(__name__)


class DataModel:
    """Datamodel for single-day ancillary data processing to NSRDB."""

    # directory to cache intermediate data (nearest neighbor results)
    CACHE_DIR = '/projects/pxs/reference_grids/_nn_query_cache'

    # source files for weight factors

    WEIGHTS: ClassVar = {
        'aod': os.path.join(
            DATADIR, 'Monthly_pixel_correction_MERRA2_AOD.txt'
        ),
        'alpha': os.path.join(
            DATADIR, 'Monthly_pixel_correction_MERRA2_Alpha.txt'
        ),
    }

    # variables used for all-sky that are processed in this module
    ALL_SKY_VARS = (
        'alpha',
        'aod',
        'asymmetry',
        'cloud_type',
        'cld_opd_dcomp',
        'cld_reff_dcomp',
        'ozone',
        'ssa',
        'surface_albedo',
        'surface_pressure',
        'total_precipitable_water',
        'solar_zenith_angle',
    )

    # variables from MERRA processed in this module
    MERRA_VARS = (
        'surface_pressure',
        'air_temperature',
        'ozone',
        'total_precipitable_water',
        'wind_speed',
        'wind_direction',
        'wind_speed_10m',
        'wind_direction_10m',
        'alpha',
        'aod',
        'ssa',
        'relative_humidity',  # Derived from MERRA vars
        'dew_point',  # Derived from MERRA vars
        'solar_zenith_angle',  # Derived from MERRA vars
    )

    # cloud variables from UW/GOES
    CLOUD_VARS = (
        'cloud_type',
        'cld_opd_dcomp',
        'cld_reff_dcomp',
        'cld_press_acha',
    )

    MLCLOUDS_VARS = (
        'cloud_fraction',
        'cloud_probability',
        'temp_3_75um_nom',
        'temp_11_0um_nom',
        'temp_11_0um_nom_stddev_3x3',
        'refl_0_65um_nom',
        'refl_0_65um_nom_stddev_3x3',
        'refl_3_75um_nom',
    )

    # all variables processed by this module
    ALL_VARS = tuple(set(ALL_SKY_VARS + MERRA_VARS + CLOUD_VARS))

    # all variables processed by this module WITH mlclouds gap fill
    ALL_VARS_ML = tuple(
        set(ALL_SKY_VARS + MERRA_VARS + CLOUD_VARS + MLCLOUDS_VARS)
    )

    def __init__(
        self,
        date,
        nsrdb_grid,
        nsrdb_freq='5min',
        var_meta=None,
        factory_kwargs=None,
        scale=True,
        max_workers=None,
    ):
        """
        Parameters
        ----------
        date : datetime.date
            Single day to extract MERRA2 data for.
        nsrdb_grid : str | pd.DataFrame
            CSV file containing the NSRDB reference grid to interpolate to,
            or a pre-extracted (and reduced) dataframe. The first csv column
            must be the NSRDB site gid's.
        nsrdb_freq : str
            Final desired NSRDB temporal frequency.
        var_meta : str | pd.DataFrame | None
            CSV file or dataframe containing meta data for all NSRDB variables.
            Defaults to the NSRDB var meta csv in git repo.
        factory_kwargs : dict | None
            Optional namespace of kwargs to use to initialize variable data
            handlers from the data model's variable factory. Keyed by
            variable name. Values can be "source_dir", "handler", etc...
            source_dir for cloud variables can be a normal directory
            path or /directory/prefix*suffix where /directory/ can have
            more sub dirs
        scale : bool
            Flag to scale source data to reduced (integer) precision after
            data model processing.
        max_workers : int | None
            Maximum number of workers to use in parallel. 1 runs serial,
            None uses all available workers.
        """

        self._nsrdb_data_shape = None
        self._nsrdb_grid_file = None
        self._date = date
        self._nsrdb_freq = nsrdb_freq
        self._var_meta = var_meta
        self._parse_nsrdb_grid(nsrdb_grid)

        self._factory_kwargs = {} if factory_kwargs is None else factory_kwargs
        self._scale = scale
        self._var_factory = VarFactory()
        self._processed = {}
        self._ti = None
        self._weights = {}
        self._max_workers = max_workers

    def __getitem__(self, key):
        return self._processed[key]

    def __setitem__(self, key, value):
        if isinstance(value, (np.ndarray, pd.DatetimeIndex, str)):
            self._processed[key] = value
        elif isinstance(value, dict):
            self._processed.update(value)
        else:
            raise TypeError(
                'Did not recognize dtype sent to DataModel '
                'processed dictionary: {}'.format(type(value))
            )

    def __contains__(self, dset):
        return dset in self._processed

    def _parse_nsrdb_grid(
        self, inp, req=('latitude', 'longitude', 'elevation')
    ):
        """Set the NSRDB reference grid from a csv file.

        Parameters
        ----------
        inp : str
            CSV file containing the NSRDB reference grid to interpolate to.
            The first column must be the NSRDB site gid's.
        req : tuple | list
            Required column labels in nsrdb grid file.
        """

        if isinstance(inp, pd.DataFrame):
            self._nsrdb_grid = clean_meta(inp)
        elif inp.endswith('.csv'):
            self._nsrdb_grid_file = inp
            self._nsrdb_grid = clean_meta(pd.read_csv(inp, index_col=0))
        else:
            raise TypeError(
                'Expected csv grid file or DataFrame but received: {}'.format(
                    inp
                )
            )

        # check requirements
        missing = [r for r in req if r not in self._nsrdb_grid]
        if any(missing):
            msg = ''
            logger.warning(msg)

        # copy index to gid data column that will be saved in output files
        # used in the case that the grid is chunked into subsets of sites
        self._nsrdb_grid['gid'] = self._nsrdb_grid.index

    @property
    def date(self):
        """Get the single-day datetime.date for this instance."""
        return self._date

    @property
    def nsrdb_grid(self):
        """Return the grid.

        Returns
        -------
        _nsrdb_grid : pd.DataFrame
            Reference grid data.
        """
        return self._nsrdb_grid

    @property
    def nsrdb_ti(self):
        """Get the NSRDB target time index.

        Returns
        -------
        nsrdb_ti : pd.DatetimeIndex
            Pandas datetime index for the current day at the NSRDB resolution.
        """
        if self._ti is None:
            self._ti = pd.date_range(
                '1-1-{y}'.format(y=self.date.year),
                '1-1-{y}'.format(y=self.date.year + 1),
                freq=self._nsrdb_freq,
            )[:-1]
            mask = (self._ti.month == self.date.month) & (
                self._ti.day == self.date.day
            )
            self._ti = self._ti[mask]
            self['time_index'] = self._ti

        return self._ti

    @property
    def nsrdb_data_shape(self):
        """Get the final NSRDB data shape for a single var.

        Returns
        -------
        _nsrdb_data_shape : tuple
            Two-entry shape tuple.
        """
        if self._nsrdb_data_shape is None:
            self._nsrdb_data_shape = (len(self.nsrdb_ti), len(self.nsrdb_grid))

        return self._nsrdb_data_shape

    @property
    def var_meta(self):
        """Get the nsrdb variables meta data table.

        Returns
        -------
        pd.DataFrame
        """
        return self._var_meta

    @property
    def processed_data(self):
        """Get the processed data dictionary.

        Returns
        -------
        _processed : dict
            Namespace of processed data set with __setitem__. Keys should be
            NSRDB variable names.
        """
        return self._processed

    def get_geo_nn(
        self,
        df1,
        df2,
        interp_method='NN',
        nn_method='haversine',
        labels=('latitude', 'longitude'),
        cache=False,
    ):
        """Get the geographic nearest neighbor distances (km) and indices.

        Parameters
        ----------
        df1/df2 : pd.DataFrame:
            Dataframes containing coodinate columns with the corresponding
            labels.
        interp_method : str
            Spatial interpolation method - either NN or IDW
        nn_method : str | None
            NSRDB nearest_neighbor tree search method, either
            "haversine" or "kdtree". None defaults to geo_nn.
        labels : tuple | list
            Column labels corresponding to the lat/lon columns in df1/df2.
        cache : bool | str
            Flag to cache nearest neighbor results or retrieve cached results
            instead of performing NN query. Strings are evaluated as the csv
            file name to cache.

        Returns
        -------
        dist : ndarray
            Distance array in km returned if return_dist input arg set to True.
        indicies : ndarray
            1D array of row indicies in df1 that match df2.
            df1[df1.index[indicies[i]]] is closest to df2[df2.index[i]]
        """

        if nn_method.lower() == 'haversine':
            nn_method = geo_nn
        elif nn_method.lower() == 'kdtree':
            nn_method = knn
        else:
            e = 'Did not recognize nn_method "{}"'.format(nn_method)
            logger.error(e)
            raise ValueError(e)

        k = (
            1
            if 'NN' in interp_method.upper()
            else 4
            if 'IDW' in interp_method.upper()
            else (parse_method(interp_method) or 4)
            if 'AGG' in interp_method.upper()
            else None
        )
        msg = f'Did not recognize spatial interp_method: "{interp_method}"'
        assert k is not None, msg

        # Do not cache results if the intended Cache directory isn't available
        if not NSRDBfs(self.CACHE_DIR).exists():
            cache = False

        if isinstance(cache, str):
            if not cache.endswith('.csv'):
                # cache file must be csv
                cache += '.csv'

            if self._nsrdb_grid_file is not None:
                # make sure cache file is nsrdb-grid-specific
                cache = cache.replace(
                    '.csv', '_' + os.path.basename(self._nsrdb_grid_file)
                )

            # try to get cached kdtree results. fast for prototyping.
            cache_d = os.path.join(
                self.CACHE_DIR, cache.replace('.csv', '_d.csv')
            )
            cache_i = os.path.join(
                self.CACHE_DIR, cache.replace('.csv', '_i.csv')
            )

            if NSRDBfs(cache_i).exists() and NSRDBfs(cache_d).exists():
                logger.warning(
                    'Found cached nearest neighbor indices, '
                    'importing: {}'.format(cache_i)
                )
                dist = np.genfromtxt(cache_d, dtype=np.float32, delimiter=',')
                ind = np.genfromtxt(cache_i, dtype=np.uint32, delimiter=',')

            else:
                dist, ind = nn_method(df1, df2, labels=labels, k=k)
                logger.info(
                    'Saving nearest neighbor indices to: {}'.format(cache_i)
                )
                np.savetxt(cache_d, dist, delimiter=',')
                np.savetxt(cache_i, ind, delimiter=',')

        else:
            dist, ind = nn_method(df1, df2, labels=labels, k=k)

        if (
            dist.shape[0] != len(df2)
            or ind.shape[0] != len(df2)
            or dist.shape[1] != k
            or ind.shape[1] != k
        ):
            e = (
                'NSRDB DataModel.get_geo_nn() method returned dist of '
                'shape {} and ind of shape {} while the query dataframe is of '
                'shape {} and k is {}. Maybe check the cached NN file.'.format(
                    dist.shape, ind.shape, df2.shape, k
                )
            )
            logger.error(e)
            raise RuntimeError(e)

        return dist, ind

    @staticmethod
    def get_cloud_nn(fp_cloud, cloud_kwargs, nsrdb_grid, dist_lim=1.0):
        """Nearest neighbors computation for cloud data regrid.

        Parameters
        ----------
        fp_cloud : str
            Single cloud source file either .nc or .h5
        cloud_kwargs : dict
            Kwargs for the initialization of CloudVarSingleH5 or
            CloudVarSingleNC along with fp_cloud
        nsrdb_grid : pd.DataFrame
            Reference grid data for NSRDB.
        dist_lim : float
            Return only neighbors within this distance during cloud regrid.
            The distance is in decimal degrees (more efficient than real
            distance). NSRDB sites further than this value from GOES data
            pixels will be warned and given missing cloud types and properties
            resulting in a full clearsky timeseries.

        Returns
        -------
        index : np.ndarray | None
            KDTree query results mapping cloud data to the NSRDB grid. e.g.
            nsrdb_data = cloud_data[index]. None if bad grid data.
        cloud_obj_single : CloudVarSingleH5 | CloudVarSingleNC
            Initialized cloud variable handler for a single cloud source file.
            The .tree property should be initialized with this return obj
        """

        cloud_obj_single = CloudVar.get_handler(fp_cloud, **cloud_kwargs)

        # Build NN tree based on the unique cloud grid at single timestep
        try:
            grid = cloud_obj_single.grid
        except Exception as e:
            msg = 'Exception building cloud NN tree for {}: {}'.format(
                cloud_obj_single, e
            )
            logger.error(msg)
            raise RuntimeError(msg) from e

        if grid is None:
            return None

        # Get the index of NN to NSRDB grid
        dist, index = cloud_obj_single.tree.query(
            nsrdb_grid[cloud_obj_single.GRID_LABELS].values, k=1
        )
        out_of_bounds = dist > dist_lim
        index[out_of_bounds] = -1

        logger.debug(
            'ReGrid distances range from {:.2f} to {:.2f} with '
            'a mean of {:.2f} and median of {:.2f} for '
            'cloud fpath: {}'.format(
                dist.min(),
                dist.max(),
                dist.mean(),
                np.median(dist),
                cloud_obj_single.fpath,
            )
        )

        if any(out_of_bounds):
            msg = (
                'The following NSRDB gids were further '
                'than {} distance from cloud coordinates: {}'.format(
                    dist_lim, np.where(out_of_bounds)[0]
                )
            )
            logger.warning(msg)
            warn(msg)

        index = index.astype(np.int32)
        cloud_obj_single.clean_attrs()

        return index, cloud_obj_single

    def get_weights(self, var_obj):
        """Get the irradiance model weights for AOD/Alpha.

        Parameters
        ----------
        var_obj : MerraVar
            Merra processing variable object.

        Returns
        -------
        weights : np.ndarray | NoneType
            1D array of weighting values for the given var in the current
            month. Returns None if var does not require weighting.
        """

        if var_obj.name in self.WEIGHTS and var_obj.name not in self._weights:
            logger.debug('Extracting weights for "{}"'.format(var_obj.name))
            wdf = pd.read_csv(
                self.WEIGHTS[var_obj.name],
                sep=' ',
                skiprows=4,
                skipinitialspace=1,
            )

            # use lat/lon and the current month and drop everything else
            current_col = str(self.date.month).zfill(2)
            wdf = wdf.rename(
                {
                    'Lat.': 'latitude',
                    'Long.': 'longitude',
                    current_col: 'weights',
                },
                axis='columns',
            )
            wdf = wdf[['latitude', 'longitude', 'weights']]

            # use geo nearest neighbors to find closest indices
            # between weights and MERRA grid
            _, i_nn = self.get_geo_nn(
                wdf, var_obj.grid, interp_method='NN', nn_method='haversine'
            )

            weight_arr = wdf['weights'].values[i_nn.ravel()]
            weight_arr[(weight_arr < 0)] = 1
            self._weights[var_obj.name] = weight_arr

        weights = self._weights.get(var_obj.name, None)

        return weights

    def scale_data(self, var, data):
        """Perform safe scaling and datatype conversion of data.

        Parameters
        ----------
        var : str
            NSRDB variable name.
        data : np.ndarray
            Data array to scale.

        Returns
        -------
        data : np.ndarray
            Scaled data array with final dtype.
        """

        if self._scale and isinstance(data, np.ndarray):
            var_obj = self._var_factory.get_base_handler(
                var, var_meta=self._var_meta, date=self.date
            )
            data = var_obj.scale_data(data)

        return data

    def unscale_data(self, var, data):
        """Perform safe un-scaling and datatype conversion of data.

        Parameters
        ----------
        var : str
            NSRDB variable name.
        data : np.ndarray
            Scaled data array to unscale.

        Returns
        -------
        data : np.ndarray
            Unscaled float32 data array.
        """

        if self._scale and isinstance(data, np.ndarray):
            var_obj = self._var_factory.get_base_handler(
                var, var_meta=self._var_meta, date=self.date
            )
            data = var_obj.unscale_data(data)

        return data

    def run_pre_flight(self, var_list):
        """Run pre-flight checks, raise if specified paths/files are not found.

        Parameters
        ----------
        var_list : list
            List of variable names
        """

        missing_list = []
        for var in var_list:
            if var in self._var_factory.MAPPING:
                var_kwargs = self._factory_kwargs.get(var, {})
                var_obj = self._var_factory.get_instance(
                    var,
                    var_meta=self._var_meta,
                    name=var,
                    date=self.date,
                    dsets=[var],
                    **var_kwargs,
                )

                if hasattr(var_obj, 'pre_flight'):
                    missing = var_obj.pre_flight()
                else:
                    missing = ''

                if missing:
                    missing_list.append(missing)

        if missing_list:
            e = (
                'The data model pre-flight checks could not find '
                'some required directories and/or files. '
                'The following are missing: {}'.format(missing_list)
            )
            logger.exception(e)
            raise OSError(e)

    @staticmethod
    def convert_units(var, data):
        """Convert MERRA data to NSRDB units.

        Parameters
        ----------
        var : str
            NSRDB Variable name.
        data : np.ndarray
            Data for var.

        Returns
        -------
        data : np.ndarray
            Data with NSRDB units if conversion is required for "var".
        """

        if var == 'air_temperature':
            if np.max(data) > 100:
                # convert Kelvin to Celsius
                data -= 273.15

        elif var == 'surface_pressure':
            if np.max(data) > 10000:
                # convert surface pressure from Pa to mbar
                data /= 100

        elif var == 'ozone':
            # convert Dobson to atm-cm
            data /= 1000.0

        elif var == 'total_precipitable_water':
            # Convert precip from kg/m2 to cm
            data *= 0.1

        return data

    @classmethod
    def check_merra_cloud_source(
        cls, var_list, cloud_vars, date, var_meta, factory_kwargs
    ):
        """Check if the cloud data source is a merra file and adjust variable
        lists and factory kwargs accordingly.

        Parameters
        ----------
        var_list : list
            List of variables being processed without the GOES cloud data
            handler
        cloud_vars : list
            List of cloud data variables from GOES being processed with the
            cloud data handler
        date : datetime.date
            Date of target processing
        var_meta : pd.DataFrame | None | str
            CSV file or dataframe containing meta data for all NSRDB variables.
        factory_kwargs : dict
            Optional namespace of kwargs to use to initialize variable data
            handlers from the data model's variable factory. Keyed by
            variable name. Values can be "source_dir", "handler", etc...
            source_dir for cloud variables can be a normal directory
            path or /directory/prefix*suffix where /directory/ can have
            more sub dirs

        Returns
        -------
        var_list : list
            List of variables being processed without the GOES cloud data
            handler - cloud variables have been added to this list if merra is
            source
        cloud_vars : list
            List of variables being processed with the GOES cloud data handler.
            This is empty if the data source is merra.
        factory_kwargs : dict
            Optional namespace of kwargs to initialize variable data. If cloud
            variables are being sourced from merra, appropriate kwargs are
            added to this dict.
        """
        merra_c_vars = (
            'cld_opd_dcomp',
            'cld_reff_dcomp',
            'cloud_type',
            'cld_press_acha',
        )

        if any(cv in cloud_vars for cv in merra_c_vars):
            var_kwargs = factory_kwargs.get('cloud_type', {})
            handler = VarFactory.get_base_handler(
                'cloud_type', var_meta=var_meta, date=date, **var_kwargs
            )
            is_merra, new_kwargs = cls.is_merra_cloud(handler)

            if is_merra:
                for var in merra_c_vars:
                    factory_kwargs[var].update(copy.deepcopy(new_kwargs))

                factory_kwargs['cld_opd_dcomp']['merra_name'] = 'TAUTOT'
                factory_kwargs['cld_opd_dcomp']['spatial_interp'] = 'IDW2'
                factory_kwargs['cld_opd_dcomp']['temporal_interp'] = 'linear'

                keep_cloud_vars = [v for v in cloud_vars if v in merra_c_vars]
                var_list += keep_cloud_vars
                cloud_vars = []

                logger.info(
                    'Updated factory kwargs for cloud data from '
                    'MERRA: {}'.format(factory_kwargs)
                )

        return var_list, cloud_vars, factory_kwargs

    @staticmethod
    def is_merra_cloud(handler):
        """Check to see if cloud variables have merra2 source files for the
        current day

        Parameters
        ----------
        handler : AncillaryVarHandler
            Base data model variable handler

        Returns
        -------
        check : bool
            True if the source is merra, False if not
        out : dict
            New factory kwargs for the variable if source is merra
        """

        pattern = handler.pattern
        if pattern is None:
            return False, {}

        if '{doy}' in pattern:
            pattern = pattern.format(doy=str(handler.doy).zfill(3))

        source_dir = os.path.dirname(pattern)
        if '*' in source_dir:
            source_dir = glob(source_dir)[0]
        if not os.path.exists(source_dir):
            return False, {}

        fns = os.listdir(source_dir)
        if not fns:
            return False, {}

        # if there is more than one source file this should be non-merra data
        if len(fns) > 1:
            return False, {}

        fns = [fn for fn in fns if fn.lower().startswith('merra')]
        if len(fns) != 1:
            return False, {}

        fn = fns[0]
        kwargs = {
            'handler': 'MerraVar',
            'pattern': os.path.join(source_dir, fn),
            'merra_dset': 'tavg1_2d_rad_Nx_clouds',
            'data_source': 'MERRA2',
            'elevation_correct': False,
            'spatial_interp': 'NN',
            'temporal_interp': 'nearest',
            'source_directory': source_dir,
        }

        if handler.name == 'cld_opd_dcomp':
            kwargs['merra_name'] = 'TAUTOT'
            kwargs['spatial_interp'] = 'IDW2'
            kwargs['temporal_interp'] = 'linear'

        return True, kwargs

    def is_cloud_var(self, var):
        """Determine whether or not the variable is a cloud variable from the
        CLAVR-x / GOES data

        Parameters
        ----------
        var : str
            NSRDB variable name

        Returns
        -------
        is_cv : bool
            True if var is a cloud variable
        """

        is_cv = var in self.CLOUD_VARS or var in self.MLCLOUDS_VARS
        var_fact_kwargs = self._factory_kwargs.get(var, {})
        if 'handler' in var_fact_kwargs:
            is_cv = var_fact_kwargs['handler'].lower() == 'cloudvar'

        return is_cv

    def is_derived_var(self, var):
        """Determine whether or not the variable is derived from primary
        source datasets

        Parameters
        ----------
        var : str
            NSRDB variable name

        Returns
        -------
        is_derived : bool
            True if var is handled using a derived variable handler
        """

        kwargs = self._factory_kwargs.get(var, {})
        VarClass = self._var_factory.get_class(
            var, var_meta=self._var_meta, **kwargs
        )
        is_derived = issubclass(VarClass, BaseDerivedVar)

        return is_derived

    def get_dependencies(self, var):
        """Get dependencies for a derived variable

        Parameters
        ----------
        var : str
            NSRDB variable name

        Returns
        -------
        deps : tuple
            Tuple of string names of dependencies of the derived variable
            input. Empty tuple if var is not derived.
        """

        kwargs = self._factory_kwargs.get(var, {})
        VarClass = self._var_factory.get_class(
            var, var_meta=self._var_meta, **kwargs
        )
        deps = ()
        if issubclass(VarClass, BaseDerivedVar):
            deps = VarClass.DEPENDENCIES

        return deps

    def init_cloud_data(self, cloud_obj_all):
        """Initialize a dictionary for all cloud datasets

        Parameters
        ----------
        cloud_obj_all : CloudVar
            Cloud variable handler that can be used to iterate over all single
            cloud file handler

        Returns
        -------
        cloud_data : dict
            Data dictionary of cloud datasets mapped to the NSRDB grid. Keys
            are the cloud variables names, values are 2D numpy arrays.
            Array shape is (n_time, n_sites).
        """

        cloud_data = {}
        for dset in cloud_obj_all._dsets:
            if dset == 'cloud_type':
                cloud_data[dset] = np.full(
                    self.nsrdb_data_shape, -15, dtype=np.int16
                )
            else:
                cloud_data[dset] = np.full(
                    self.nsrdb_data_shape, np.nan, dtype=np.float32
                )
        return cloud_data

    @classmethod
    def get_single_cloud_data(
        cls, fp_cloud, cloud_kwargs, nsrdb_grid, dist_lim=1.0
    ):
        """Get all that good stuff from a cloud data file.

        Parameters
        ----------
        fp_cloud : str
            Single cloud source file either .nc or .h5
        cloud_kwargs : dict
            Kwargs for the initialization of CloudVarSingleH5 or
            CloudVarSingleNC along with fp_cloud
        nsrdb_grid : pd.DataFrame
            Reference grid data for NSRDB.
        dist_lim : float
            Return only neighbors within this distance during cloud regrid.
            The distance is in decimal degrees (more efficient than real
            distance). NSRDB sites further than this value from GOES data
            pixels will be warned and given missing cloud types and properties
            resulting in a full clearsky timeseries.

        Returns
        -------
        single_data : dict | None
            Dictionary of source data for the single cloud file (single
            timestep) mapped onto the nsrdb_grid. Keys are nsrdb cloud dataset
            names and values are 1D (space,) arrays of data matching the
            nsrdb_grid. Returns None if something went wrong.
        """

        regrid_ind, cloud_obj_single = cls.get_cloud_nn(
            fp_cloud, cloud_kwargs, nsrdb_grid, dist_lim=dist_lim
        )

        single_data = None
        if cloud_obj_single is not None:
            assert regrid_ind is not None
            single_data = {}
            out_of_bounds = regrid_ind < 0
            for dset, array in cloud_obj_single.source_data.items():
                if array.size:
                    # write single timestep with NSRDB sites to appropriate row
                    # map the regridded single_data using the regrid NN indices
                    single_data[dset] = array[regrid_ind]

                else:
                    # if cloud data array has zero size, something about the
                    # cloud file was corrupted and returned no data
                    msg = (
                        'Cloud dataset "{}" had no valid data for source '
                        'file: {}'.format(dset, os.path.basename(fp_cloud))
                    )
                    logger.warning(msg)
                    warn(msg)
                    if dset == 'cloud_type':
                        single_data[dset] = np.full(
                            len(regrid_ind), -15, dtype=np.int16
                        )
                    else:
                        single_data[dset] = np.full(
                            len(regrid_ind), np.nan, dtype=np.float32
                        )

                if any(out_of_bounds):
                    if dset == 'cloud_type':
                        single_data[dset][out_of_bounds] = -15
                    else:
                        single_data[dset][out_of_bounds] = np.nan

        # try to free up mem usage.
        cloud_obj_single.clean_attrs()
        del cloud_obj_single

        return single_data

    def _cloud_regrid_serial(self, cloud_obj_all, dist_lim=1.0):
        """Perform the ReGrid nearest neighbor calculations in serial.

        Parameters
        ----------
        cloud_obj_all : CloudVar
            Cloud variable handler that can be used to iterate over all single
            cloud file handler
        dist_lim : float
            Return only neighbors within this distance during cloud regrid.
            The distance is in decimal degrees (more efficient than real
            distance). NSRDB sites further than this value from GOES data
            pixels will be warned and given missing cloud types and properties
            resulting in a full clearsky timeseries.

        Returns
        -------
        cloud_data : dict
            Data dictionary of cloud datasets mapped to the NSRDB grid. Keys
            are the cloud variables names, values are 2D numpy arrays.
            Array shape is (n_time, n_sites).
        """
        cloud_data = self.init_cloud_data(cloud_obj_all)

        cloud_kwargs = cloud_obj_all.single_handler_kwargs

        for i, (index, row) in enumerate(cloud_obj_all.file_df.iterrows()):
            assert index == self.nsrdb_ti[i]
            fp_cloud = row['flist']
            fp_cloud_msg = (
                fp_cloud
                if not isinstance(fp_cloud, str)
                else os.path.basename(fp_cloud)
            )
            mem = psutil.virtual_memory()
            logger.info(
                'Cloud data timestep {} has source file: {}. '
                'Memory usage is {:.3f} GB out of '
                '{:.3f} GB total.'.format(
                    index, fp_cloud_msg, mem.used / 1e9, mem.total / 1e9
                )
            )
            if isinstance(fp_cloud, str):
                single_data = self.get_single_cloud_data(
                    fp_cloud, cloud_kwargs, self.nsrdb_grid, dist_lim=dist_lim
                )

                for dset, array in single_data.items():
                    # write single timestep with NSRDB sites to row
                    cloud_data[dset][i, :] = array

        return cloud_data

    def _cloud_regrid_parallel(
        self, cloud_obj_all, dist_lim=1.0, max_workers=None
    ):
        """Perform the ReGrid nearest neighbor calculations in parallel.

        Parameters
        ----------
        cloud_obj_all : CloudVar
            Cloud variable handler that can be used to iterate over all single
            cloud file handler
        dist_lim : float
            Return only neighbors within this distance during cloud regrid.
            The distance is in decimal degrees (more efficient than real
            distance). NSRDB sites further than this value from GOES data
            pixels will be warned and given missing cloud types and properties
            resulting in a full clearsky timeseries.
        max_workers : None | int
            Max parallel workers allowed for regrid processing. None uses all
            available workers. 1 runs regrid in serial.

        Returns
        -------
        cloud_data : dict
            Data dictionary of cloud datasets mapped to the NSRDB grid. Keys
            are the cloud variables names, values are 2D numpy arrays.
            Array shape is (n_time, n_sites).
        """

        logger.debug('Starting cloud Regrid and IO in parallel.')

        cloud_data = self.init_cloud_data(cloud_obj_all)

        # start a local cluster
        logger.debug(
            'Starting local cluster with {} workers.'.format(max_workers)
        )
        futures = {}
        loggers = ['nsrdb']
        cloud_kwargs = cloud_obj_all.single_handler_kwargs
        with SpawnProcessPool(loggers=loggers, max_workers=max_workers) as exe:
            # make the nearest neighbors regrid index mapping for all timesteps
            for i, (index, row) in enumerate(cloud_obj_all.file_df.iterrows()):
                assert index == self.nsrdb_ti[i]
                fp_cloud = row['flist']
                fp_cloud_msg = (
                    fp_cloud
                    if not isinstance(fp_cloud, str)
                    else os.path.basename(fp_cloud)
                )
                logger.info(
                    'Cloud data timestep {} has source file: {}.'.format(
                        index, fp_cloud_msg
                    )
                )
                if isinstance(fp_cloud, str):
                    future = exe.submit(
                        self.get_single_cloud_data,
                        fp_cloud,
                        cloud_kwargs,
                        self.nsrdb_grid,
                        dist_lim=dist_lim,
                    )
                    futures[future] = i

            mem = psutil.virtual_memory()

            logger.info(
                'All {} cloud data futures submitted! '
                'Memory usage is {:.3f} GB out of {:.3f} GB total.'.format(
                    len(futures), mem.used / 1e9, mem.total / 1e9
                )
            )

            for completed, future in enumerate(as_completed(futures)):
                i = futures[future]
                single_data = future.result()

                for dset, array in single_data.items():
                    # write single timestep with NSRDB sites to row
                    cloud_data[dset][i, :] = array

                mem = psutil.virtual_memory()
                logger.info(
                    'Cloud data Regrid and IO futures completed: '
                    '{} out of {}. Current memory usage is '
                    '{:.3f} GB out of {:.3f} GB total.'.format(
                        completed,
                        len(futures),
                        mem.used / 1e9,
                        mem.total / 1e9,
                    )
                )

        return cloud_data

    def _process_clouds(self, cloud_vars, dist_lim=1.0, max_workers=None):
        """Process data for multiple cloud variables to the NSRDB grid. This
        has two main steps: 1) the regrid process which maps the cloud
        coordinates to the NSRDB meta data using parallel KDTrees, and 2) the
        cloud data IO which reads the separate cloud data files in parallel.

        (most efficient to process all cloud variables together to minimize
        number of kdtrees during regrid)

        Parameters
        ----------
        cloud_vars : tuple | list
            Source datasets to extract. It is more efficient to extract all
            required datasets at once from each cloud file, so that only one
            kdtree is built for each unique coordinate set in each cloud file.
        dist_lim : float
            Return only neighbors within this distance during cloud regrid.
            The distance is in decimal degrees (more efficient than real
            distance). NSRDB sites further than this value from GOES data
            pixels will be warned and given missing cloud types and properties
            resulting in a full clearsky timeseries.
        max_workers : None | int
            Max parallel workers allowed for cloud regrid processing. None uses
            all available workers. 1 runs regrid in serial.

        Returns
        -------
        cloud_data : dict
            Data dictionary of cloud datasets mapped to the NSRDB grid. Keys
            are the cloud variables names, values are 2D numpy arrays.
            Array shape is (n_time, n_sites).
        """

        # use the first cloud var name to get object,
        # full cloud_var list is passed in kwargs
        logger.info('Starting DataModel Cloud Regrid / IO process')

        # TODO this is imprecise getting the kwargs for just the first cloud
        # variable to be used for all the cloud variable objects... But also
        # using a single custom cloud variable name to key the cloud kwargs
        # seems too implicit. Not sure if there's a better solution here. At
        # least logging this will help the user debug if something goes wrong.
        var_kwargs = self._factory_kwargs.get(cloud_vars[0], {})
        logger.info(
            'Variable factory kwargs for cloud data processing: {}'.format(
                var_kwargs
            )
        )

        cloud_obj_all = self._var_factory.get_instance(
            cloud_vars[0],
            var_meta=self._var_meta,
            name=cloud_vars[0],
            date=self.date,
            dsets=cloud_vars,
            freq=self.nsrdb_ti.freqstr,
            **var_kwargs,
        )

        logger.debug(
            'Cloud ReGrid file list of length {}: \n\t{}'.format(
                len(cloud_obj_all.flist), '\n\t'.join(cloud_obj_all.flist)
            )
        )

        # cloud regrid
        if max_workers != 1:
            logger.info(
                'Starting cloud data Regrid and IO with {} futures '
                '(cloud timesteps) with {} parallel workers.'.format(
                    len(cloud_obj_all.flist), max_workers
                )
            )

            cloud_data = self._cloud_regrid_parallel(
                cloud_obj_all, dist_lim=dist_lim, max_workers=max_workers
            )

        else:
            logger.info(
                'Starting cloud data Regrid and IO with {} iterations '
                '(cloud timesteps) in serial.'.format(len(cloud_obj_all.flist))
            )
            cloud_data = self._cloud_regrid_serial(
                cloud_obj_all, dist_lim=dist_lim
            )

        logger.debug('Finished processing ReGrid and cloud data extraction.')

        # scale if requested
        if self._scale:
            for var, arr in cloud_data.items():
                cloud_data[var] = self.scale_data(var, arr)

        logger.info(
            'Finished extracting cloud data and writing to NSRDB arrays.'
        )
        mem = psutil.virtual_memory()
        logger.info(
            'Current memory usage is {:.3f} GB out of {:.3f} GB total.'.format(
                mem.used / 1e9, mem.total / 1e9
            )
        )

        return cloud_data

    def _process_dependencies(self, dependencies):
        """Ensure that all dependencies have been processed and set to self.

        Parameters
        ----------
        dependencies : list | tuple
            List of variable names representing dependencies that have to be
            processed before a downstream variable.
        """

        for dep in dependencies:
            # process and save data to processed attribute
            # (unscale to physical units)
            if dep not in self._processed:
                logger.info('Processing dependency "{}".'.format(dep))
                self[dep] = self._interpolate(dep)
                self[dep] = self.unscale_data(dep, self[dep])

            # dependency data dumped to disk, load from disk
            elif isinstance(self._processed[dep], str):
                logger.debug(
                    'Importing dependency "{}" from: {}'.format(
                        dep, self._processed[dep]
                    )
                )
                with Outputs(self._processed[dep]) as dep_out:
                    self[dep] = dep_out[dep]

            # dependency already in memory. Ensure physical units.
            else:
                self[dep] = self.unscale_data(dep, self[dep])

    def _derive(self, var, fpath_out=None):
        """Method for deriving variables (with dependencies).

        Parameters
        ----------
        var : str
            NSRDB var name.
        fpath_out : str | None
            File path to dump results. If no file path is given, results will
            be returned as an object.

        Returns
        -------
        data : np.ndarray
            NSRDB-resolution data for the given var and the current day.
        """

        if fpath_out is not None:
            fpath_out = fpath_out.format(var=var, i=self.nsrdb_grid.index[0])
            if NSRDBfs(fpath_out).exists():
                logger.info(
                    'Skipping DataModel for "{}" with existing '
                    'fpath_out: {}'.format(var, fpath_out)
                )
                return fpath_out
            logger.info(
                'Processing DataModel for "{}" with fpath_out: {}'.format(
                    var, fpath_out
                )
            )

        # ensure dependencies are processed before working on derived var
        dependencies = self.get_dependencies(var)
        self._process_dependencies(dependencies)

        # get the derivation object from the var factory
        kwargs = self._factory_kwargs.get(var, {})
        obj = self._var_factory.get_instance(var, **kwargs)

        try:
            dep_kwargs = {k: self[k] for k in dependencies}
            if var == 'solar_zenith_angle':
                data = obj.derive(
                    self.nsrdb_ti,
                    self.nsrdb_grid[['latitude', 'longitude']].values,
                    self.nsrdb_grid['elevation'].values,
                    **dep_kwargs,
                )
            else:
                data = obj.derive(**dep_kwargs)

        except Exception as e:
            logger.exception(
                'Could not derive "{}" using "{}", received the '
                'exception: {}'.format(var, obj, e)
            )
            raise e

        # convert units from MERRA to NSRDB
        data = self.convert_units(var, data)

        # scale if requested
        data = self.scale_data(var, data)

        # re-scale dependencies
        # (they had to be previously unscaled for physical eqns)
        for dep in dependencies:
            self[dep] = self.scale_data(dep, self[dep])

        if fpath_out is not None:
            data = self.dump(var, fpath_out, data, purge=True)

        logger.info('Finished "{}".'.format(var))

        return data

    def _interpolate(self, var):
        """Method for processing interpolated variables.

        Parameters
        ----------
        var : str
            NSRDB var name.

        Returns
        -------
        data : np.ndarray
            NSRDB-resolution data for the given var and the current day.
        """

        var_kwargs = self._factory_kwargs.get(var, {})
        var_obj = self._var_factory.get_instance(
            var,
            var_meta=self._var_meta,
            name=var,
            date=self.date,
            **var_kwargs,
        )

        if 'albedo' in var:
            # special exclusions for large-extent albedo
            var_obj.exclusions_from_nsrdb(self.nsrdb_grid)

        # get ancillary data source data array
        data = var_obj.source_data

        # get mapping from source data grid to NSRDB
        dist, ind = self.get_geo_nn(
            var_obj.grid,
            self.nsrdb_grid,
            interp_method=var_obj.spatial_method,
            nn_method=var_obj.NN_METHOD,
            cache=var_obj.cache_file,
        )

        # perform weighting if applicable
        if var in self.WEIGHTS and 'merra' in str(type(var_obj)).lower():
            weights = self.get_weights(var_obj)
            if weights is not None:
                logger.debug('Applying weights to "{}".'.format(var))
                data *= weights

        # run spatial interpolation
        logger.debug(
            'Performing spatial interpolation on "{}" with method "{}", '
            'metric "{}", and with shape {}'.format(
                var, var_obj.spatial_method, var_obj.NN_METHOD, data.shape
            )
        )
        data = spatial_interp(
            var,
            data,
            var_obj.grid,
            self.nsrdb_grid,
            var_obj.spatial_method,
            dist,
            ind,
            var_obj.elevation_correct,
        )

        # run temporal interpolation
        if var_obj.temporal_method == 'linear':
            logger.debug(
                'Performing linear temporal interpolation on '
                '"{}" with shape {}'.format(var, data.shape)
            )
            data = temporal_lin(data, var_obj.time_index, self.nsrdb_ti)

        elif var_obj.temporal_method == 'nearest':
            logger.debug(
                'Performing stepwise temporal interpolation on '
                '"{}" with shape {}'.format(var, data.shape)
            )
            data = temporal_step(data, var_obj.time_index, self.nsrdb_ti)

        # convert units from MERRA to NSRDB
        data = self.convert_units(var, data)

        # scale if requested
        data = self.scale_data(var, data)

        return data

    @classmethod
    def _get_var_lists(cls, var_list, data_model):
        # default multiple compute
        if var_list is None:
            var_list = cls.ALL_VARS

        deps = ()
        for var in var_list:
            deps += data_model.get_dependencies(var)

        var_list += deps
        var_list = tuple(set(var_list))

        # remove cloud variables from var_list to be processed together
        # (most efficient to process all cloud variables together to minimize
        # number of kdtrees during regrid)
        cloud_vars = [var for var in var_list if data_model.is_cloud_var(var)]
        var_list = [v for v in var_list if v not in cloud_vars]

        # remove derived (dependent) variables from var_list to be processed
        # last (most efficient to process depdencies first, dependents last)
        derived_vars = [v for v in var_list if data_model.is_derived_var(v)]
        var_list = [v for v in var_list if v not in derived_vars]

        return var_list, cloud_vars, derived_vars

    @classmethod
    def _process_multiple(
        cls,
        var_list,
        date,
        nsrdb_grid,
        nsrdb_freq='5min',
        dist_lim=1.0,
        var_meta=None,
        max_workers=None,
        max_workers_regrid=None,
        scale=True,
        fpath_out=None,
        factory_kwargs=None,
    ):
        """Process ancillary data for multiple variables for a single day.

        Parameters
        ----------
        var_list : list | None
            List of variables to process
        date : datetime.date
            Single day to extract ancillary data for.
        nsrdb_grid : str | pd.DataFrame
            CSV file containing the NSRDB reference grid to interpolate to,
            or a pre-extracted (and reduced) dataframe.
        nsrdb_freq : str
            Final desired NSRDB temporal frequency.
        dist_lim : float
            Return only neighbors within this distance during cloud regrid.
            The distance is in decimal degrees (more efficient than real
            distance). NSRDB sites further than this value from GOES data
            pixels will be warned and given missing cloud types and properties
            resulting in a full clearsky timeseries.
        var_meta : str | pd.DataFrame
            CSV file or dataframe containing meta data for all NSRDB variables.
        max_workers : int | None
            Maximum workers to use in parallel. 1 will run serial, None will
            use all available parallel workers.
        max_workers_regrid : None | int
            Max parallel workers allowed for cloud regrid processing. None uses
            all available workers. 1 runs regrid in serial.
        return_obj : bool
            Flag to return full DataModel object instead of just the processed
            data dictionary.
        fpath_out : str | None
            File path to dump results. If no file path is given, results will
            be returned as an object.
        scale : bool
            Flag to scale source data to reduced (integer) precision after
            data model processing.
        factory_kwargs : dict | None
            Optional namespace of kwargs to use to initialize variable data
            handlers from the data model's variable factory. Keyed by
            variable name. Values can be "source_dir", "handler", etc...
            source_dir for cloud variables can be a normal directory
            path or /directory/prefix*suffix where /directory/ can have
            more sub dirs

        Returns
        -------
        out : DataModel
            Full DataModel object with the data in the .processed property.
        """

        data_model = cls(
            date,
            nsrdb_grid,
            nsrdb_freq=nsrdb_freq,
            var_meta=var_meta,
            factory_kwargs=factory_kwargs,
            scale=scale,
            max_workers=max_workers,
        )

        var_list, cloud_vars, derived_vars = cls._get_var_lists(
            var_list=var_list, data_model=data_model
        )

        temp = cls.check_merra_cloud_source(
            var_list, cloud_vars, date, var_meta, factory_kwargs
        )
        var_list, cloud_vars, factory_kwargs = temp

        factory_kwargs = {} if factory_kwargs is None else factory_kwargs
        data_model._factory_kwargs = factory_kwargs

        # run pre-flight checks
        data_model.run_pre_flight(var_list)
        data_model.run_pre_flight(cloud_vars)
        data_model.run_pre_flight(derived_vars)

        logger.info(
            f'First processing data for variable list: {var_list}. Then '
            f'processing cloud data for variable list: {cloud_vars}. Finally, '
            f'processing derived data for variable list: {derived_vars}'
        )

        if var_list and max_workers == 1:
            data = {}
            for var in var_list:
                data_model[var] = cls.run_single(
                    var,
                    date,
                    nsrdb_grid,
                    nsrdb_freq=nsrdb_freq,
                    var_meta=var_meta,
                    fpath_out=fpath_out,
                    scale=scale,
                    factory_kwargs=factory_kwargs,
                )

        # run in parallel
        elif var_list:
            data = cls._process_parallel(
                var_list,
                date,
                nsrdb_grid,
                max_workers=max_workers,
                nsrdb_freq=nsrdb_freq,
                var_meta=var_meta,
                fpath_out=fpath_out,
                scale=scale,
                factory_kwargs=factory_kwargs,
            )

            for k, v in data.items():
                data_model[k] = v

        # process cloud variables together
        if cloud_vars:
            data_model['clouds'] = cls.run_clouds(
                cloud_vars,
                date,
                nsrdb_grid,
                nsrdb_freq=nsrdb_freq,
                dist_lim=dist_lim,
                var_meta=var_meta,
                max_workers_regrid=max_workers_regrid,
                fpath_out=fpath_out,
                scale=scale,
                factory_kwargs=factory_kwargs,
            )

        # process derived (dependent) variables last using the built
        # AncillaryDataProcessing object instance.
        if derived_vars:
            for var in derived_vars:
                data_model[var] = data_model._derive(var, fpath_out=fpath_out)

        # scale if requested
        if scale:
            for var, arr in data_model._processed.items():
                data_model[var] = data_model.scale_data(var, arr)

        return data_model

    @classmethod
    def _process_parallel(
        cls, var_list, date, nsrdb_grid, max_workers=None, scale=True, **kwargs
    ):
        """Process ancillary variables in parallel.

        Parameters
        ----------
        var_list : list | tuple
            List of variables to process in parallel
        date : datetime.date
            Single day to extract MERRA2 data for.
        nsrdb_grid : str | pd.DataFrame
            CSV file containing the NSRDB reference grid to interpolate to,
            or a pre-extracted (and reduced) dataframe.
        max_workers : int | None
            Optional limit for maximum parallel workers.
            None will use all available.
        scale : bool
            Flag to scale source data to reduced (integer) precision after
            data model processing.
        **kwargs : dict
            Keyword args to pass to DataModel.run_single.

        Returns
        -------
        futures : dict
            Gathered futures, namespace of nsrdb data numpy arrays keyed by
            nsrdb variable name.
        """

        var_list = list(set(var_list))
        logger.info('Processing variables in parallel: {}'.format(var_list))
        # start a local cluster
        futures = {}
        logger.debug(
            'Starting local cluster with {} workers.'.format(max_workers)
        )
        loggers = ['nsrdb']
        with SpawnProcessPool(loggers=loggers, max_workers=max_workers) as exe:
            # submit a future for each merra variable (non-calculated)
            for var in var_list:
                futures[var] = exe.submit(
                    cls.run_single,
                    var,
                    date,
                    nsrdb_grid,
                    scale=scale,
                    **kwargs,
                )

            # watch memory during futures to get max memory usage
            logger.debug('Waiting on parallel futures...')
            max_mem = 0
            running = len(futures)
            while running > 0:
                mem = psutil.virtual_memory()
                max_mem = np.max((mem.used / 1e9, max_mem))
                time.sleep(5)
                running = 0
                keys = []
                for key, future in futures.items():
                    if future.running():
                        running += 1
                        keys += [key]

                logger.info(
                    '{} DataModel processing futures are running: {} '
                    'memory usage is {:.3f} GB out of {:.3f} GB total'.format(
                        running, keys, mem.used / 1e9, mem.total / 1e9
                    )
                )

            logger.info(
                'Futures finished, maximum memory usage was '
                '{:.3f} GB out of {:.3f} GB total.'.format(
                    max_mem, mem.total / 1e9
                )
            )

            mem = psutil.virtual_memory()
            logger.info(
                'Current memory usage is '
                '{:.3f} GB out of {:.3f} GB total.'.format(
                    mem.used / 1e9, mem.total / 1e9
                )
            )

            # gather results
            for k, v in futures.items():
                futures[k] = v.result()

        return futures

    def dump(self, var, fpath_out, data, purge=False, mode='w'):
        """Run ancillary data processing for one variable for a single day.

        Parameters
        ----------
        var : str
            NSRDB var name.
        fpath_out : str | None
            File path to dump results. If no file path is given, results will
            be returned as an object.
        data : np.ndarray
            NSRDB-resolution data for the given var and the current day.
        purge : bool
            Flag to purge data from memory after dumping to disk
        mode : str, optional
            Mode to open fpath_out with, by default 'w'

        Returns
        -------
        data : str | np.ndarray
            Input data array if no purge, else file path to dump results.
        """

        if isinstance(fpath_out, str):
            if '{var}' in fpath_out and '{i}' in fpath_out:
                fpath_out = fpath_out.format(
                    var=var, i=self.nsrdb_grid.index[0]
                )

            logger.info(
                'Writing {} to: {}'.format(var, os.path.basename(fpath_out))
            )

            if data is None:
                data = self[var]

            # make file for each var
            fpath_out += '.tmp'
            with Outputs(fpath_out, mode=mode) as fout:
                if 'time_index' not in fout:
                    fout.time_index = self.nsrdb_ti

                if 'meta' not in fout:
                    fout.meta = self.nsrdb_grid

                var_kwargs = self._factory_kwargs.get(var, {})
                var_obj = VarFactory.get_base_handler(
                    var, var_meta=self._var_meta, date=self.date, **var_kwargs
                )
                attrs = var_obj.attrs

                fout._add_dset(
                    dset_name=var,
                    data=data,
                    dtype=var_obj.final_dtype,
                    chunks=var_obj.chunks,
                    attrs=attrs,
                )

            os.rename(fpath_out, fpath_out.replace('.tmp', ''))
            fpath_out = fpath_out.replace('.tmp', '')
            if purge:
                del data
                data = fpath_out

        return data

    @classmethod
    def run_single(
        cls,
        var,
        date,
        nsrdb_grid,
        nsrdb_freq='5min',
        var_meta=None,
        fpath_out=None,
        factory_kwargs=None,
        scale=True,
        **kwargs,
    ):
        """Run ancillary data processing for one variable for a single day.

        Parameters
        ----------
        var : str
            NSRDB var name.
        date : datetime.date
            Single day to extract ancillary data for.
        nsrdb_grid : str | pd.DataFrame
            CSV file containing the NSRDB reference grid to interpolate to,
            or a pre-extracted (and reduced) dataframe. The first csv column
            must be the NSRDB site gid's.
        nsrdb_freq : str
            Final desired NSRDB temporal frequency.
        var_meta : str | pd.DataFrame | None
            CSV file or dataframe containing meta data for all NSRDB variables.
            Defaults to the NSRDB var meta csv in git repo.
        fpath_out : str | None
            File path to dump results. If no file path is given, results will
            be returned as an object.
        factory_kwargs : dict | None
            Optional namespace of kwargs to use to initialize variable data
            handlers from the data model's variable factory. Keyed by
            variable name. Values can be "source_dir", "handler", etc...
            source_dir for cloud variables can be a normal directory
            path or /directory/prefix*suffix where /directory/ can have
            more sub dirs
        scale : bool
            Flag to scale source data to reduced (integer) precision after
            data model processing.
        kwargs : dict
            Optional kwargs. Based on the NSRDB var name requested to be
            processed, this method runs one of several DataModel processing
            methods (_interpolate, _derive, _process_clouds). These kwargs will
            get passed to the processing method.

        Returns
        -------
        data : np.ndarray
            NSRDB-resolution data for the given var and the current day.
        """

        data_model = cls(
            date,
            nsrdb_grid,
            nsrdb_freq=nsrdb_freq,
            var_meta=var_meta,
            factory_kwargs=factory_kwargs,
            scale=scale,
        )

        if fpath_out is None:
            logger.info('Processing data for "{}".'.format(var))
        else:
            if '{var}' not in fpath_out or '{i}' not in fpath_out:
                raise OSError(
                    'Cannot write to fpath_out, need "var" and "i" '
                    'format keywords: {}'.format(fpath_out)
                )
            fpath_out = fpath_out.format(
                var=var, i=data_model.nsrdb_grid.index[0]
            )
            if NSRDBfs(fpath_out).exists():
                logger.info(
                    'Skipping DataModel for "{}" with existing '
                    'fpath_out: {}'.format(var, fpath_out)
                )
                return fpath_out
            logger.info(
                'Processing DataModel for "{}" with fpath_out: {}'.format(
                    var, fpath_out
                )
            )

        if data_model.is_cloud_var(var):
            method = data_model._process_clouds
        elif data_model.is_derived_var(var):
            method = data_model._derive
        else:
            method = data_model._interpolate

        try:
            data = method(var, **kwargs)
        except Exception as e:
            logger.exception(
                'Processing method "DataModel.{}()" failed for "{}"'.format(
                    method.__name__, var
                )
            )
            raise e

        if data.shape != data_model.nsrdb_data_shape:
            raise ValueError(
                'Expected NSRDB data shape of {}, but '
                'received shape {} for "{}"'.format(
                    data_model.nsrdb_data_shape, data.shape, var
                )
            )

        if fpath_out is not None:
            data = data_model.dump(var, fpath_out, data, purge=True)

        logger.info('Finished "{}".'.format(var))

        return data

    @classmethod
    def _check_cloud_output(cls, data, data_model):
        for k, v in data.items():
            if v.shape != data_model.nsrdb_data_shape:
                raise ValueError(
                    'Expected NSRDB data shape of {}, but received shape {} '
                    'for "{}"'.format(data_model.nsrdb_data_shape, v.shape, k)
                )

    @classmethod
    def _write_cloud_output(cls, data, data_model, fpath_out):
        if fpath_out is not None:
            for var, arr in data.items():
                fpath_out_var = fpath_out.format(
                    var=var, i=data_model.nsrdb_grid.index[0]
                )
                data[var] = data_model.dump(
                    var, fpath_out_var, arr, purge=True
                )

    @classmethod
    def run_clouds(
        cls,
        cloud_vars,
        date,
        nsrdb_grid,
        nsrdb_freq='5min',
        dist_lim=1.0,
        var_meta=None,
        max_workers_regrid=None,
        scale=True,
        fpath_out=None,
        factory_kwargs=None,
    ):
        """Run cloud processing for multiple cloud variables.

        (most efficient to process all cloud variables together to minimize
        number of kdtrees during regrid)

        Parameters
        ----------
        cloud_vars : list | tuple
            NSRDB cloud variables names.
        date : datetime.date
            Single day to extract ancillary data for.
        nsrdb_grid : str | pd.DataFrame
            CSV file containing the NSRDB reference grid to interpolate to,
            or a pre-extracted (and reduced) dataframe. The first csv column
            must be the NSRDB site gid's.
        nsrdb_freq : str
            Final desired NSRDB temporal frequency.
        dist_lim : float
            Return only neighbors within this distance during cloud regrid.
            The distance is in decimal degrees (more efficient than real
            distance). NSRDB sites further than this value from GOES data
            pixels will be warned and given missing cloud types and properties
            resulting in a full clearsky timeseries.
        var_meta : str | pd.DataFrame | None
            CSV file or dataframe containing meta data for all NSRDB variables.
            Defaults to the NSRDB var meta csv in git repo.
        max_workers_regrid : None | int
            Max parallel workers allowed for cloud regrid processing. None uses
            all available workers. 1 runs regrid in serial.
        scale : bool
            Flag to scale source data to reduced (integer) precision after
            data model processing.
        fpath_out : str | None
            File path to dump results. If no file path is given, results will
            be returned as an object.
        factory_kwargs : dict | None
            Optional namespace of kwargs to use to initialize variable data
            handlers from the data model's variable factory. Keyed by
            variable name. Values can be "source_dir", "handler", etc...
            source_dir for cloud variables can be a normal directory
            path or /directory/prefix*suffix where /directory/ can have
            more sub dirs

        Returns
        -------
        data : dict
            Namespace of nsrdb data numpy arrays keyed by nsrdb variable name.
        """

        logger.info(
            f'Processing data for multiple cloud variables: {cloud_vars}'
        )

        data_model = cls(
            date,
            nsrdb_grid,
            nsrdb_freq=nsrdb_freq,
            var_meta=var_meta,
            factory_kwargs=factory_kwargs,
            max_workers=max_workers_regrid,
            scale=scale,
        )

        if fpath_out is not None:
            data = {}
            skip_list = []
            for var in cloud_vars:
                fpath_out_var = fpath_out.format(
                    var=var, i=data_model.nsrdb_grid.index[0]
                )
                if NSRDBfs(fpath_out_var).exists():
                    logger.info(
                        'Skipping DataModel for "{}" with existing '
                        'fpath_out: {}'.format(var, fpath_out_var)
                    )
                    skip_list.append(var)
                    data[var] = fpath_out_var

            if any(skip_list):
                cloud_vars = [cv for cv in cloud_vars if cv not in skip_list]

            if not any(cloud_vars):
                return data

        try:
            data = data_model._process_clouds(
                cloud_vars, max_workers=max_workers_regrid, dist_lim=dist_lim
            )
        except Exception as e:
            logger.exception(
                'Processing method "DataModel._process_clouds()" '
                'failed for "{}"'.format(cloud_vars)
            )
            raise e

        cls._check_cloud_output(data=data, data_model=data_model)

        cls._write_cloud_output(
            data=data, data_model=data_model, fpath_out=fpath_out
        )

        logger.info('Finished "{}".'.format(cloud_vars))

        return data

    @classmethod
    def run_multiple(
        cls,
        var_list,
        date,
        nsrdb_grid,
        nsrdb_freq='5min',
        dist_lim=1.0,
        var_meta=None,
        max_workers=None,
        max_workers_regrid=None,
        return_obj=False,
        scale=True,
        fpath_out=None,
        factory_kwargs=None,
    ):
        """Run ancillary data processing for multiple variables for single day.

        Parameters
        ----------
        var_list : list | None
            List of variables to process
        date : datetime.date
            Single day to extract ancillary data for.
        nsrdb_grid : str | pd.DataFrame
            CSV file containing the NSRDB reference grid to interpolate to,
            or a pre-extracted (and reduced) dataframe. The first csv column
            must be the NSRDB site gid's.
        nsrdb_freq : str
            Final desired NSRDB temporal frequency.
        dist_lim : float
            Return only neighbors within this distance during cloud regrid.
            The distance is in decimal degrees (more efficient than real
            distance). NSRDB sites further than this value from GOES data
            pixels will be warned and given missing cloud types and properties
            resulting in a full clearsky timeseries.
        var_meta : str | pd.DataFrame | None
            CSV file or dataframe containing meta data for all NSRDB variables.
            Defaults to the NSRDB var meta csv in git repo.
        max_workers : int | None
            Number of workers to run in parallel. 1 will run serial,
            None will use all available.
        max_workers_regrid : None | int
            Max parallel workers allowed for cloud regrid processing. None uses
            all available workers. 1 runs regrid in serial.
        return_obj : bool
            Flag to return full DataModel object instead of just the processed
            data dictionary.
        scale : bool
            Flag to scale source data to reduced (integer) precision after
            data model processing.
        fpath_out : str | None
            File path to dump results. If no file path is given, results will
            be returned as an object.
        factory_kwargs : dict | None
            Optional namespace of kwargs to use to initialize variable data
            handlers from the data model's variable factory. Keyed by
            variable name. Values can be "source_dir", "handler", etc...
            source_dir for cloud variables can be a normal directory
            path or /directory/prefix*suffix where /directory/ can have
            more sub dirs

        Returns
        -------
        out : dict | DataModel
            Either the dictionary of data or the full DataModel object with
            the data in the .processed property. Controlled by the return_obj
            flag.
        """

        logger.info(
            'Building NSRDB data model for {} at a {} temporal '
            'resolution.'.format(date, nsrdb_freq)
        )

        if isinstance(nsrdb_grid, str):
            logger.info(
                'Using the NSRDB reference grid file: {}'.format(nsrdb_grid)
            )
        elif isinstance(nsrdb_grid, pd.DataFrame):
            logger.info(
                'Using the NSRDB reference grid dataframe with '
                'shape, head, tail:\n{}\n{}\n{}'.format(
                    nsrdb_grid.shape, nsrdb_grid.head(), nsrdb_grid.tail()
                )
            )
        else:
            raise TypeError(
                'Expected csv grid file or DataFrame but received: {}'.format(
                    nsrdb_grid
                )
            )

        logger.debug(
            'Running DataModel with the following var meta: {}'.format(
                var_meta
            )
        )
        logger.debug(
            'Running DataModel with the following variable '
            'factory kwargs: {}'.format(factory_kwargs)
        )

        data_model = cls._process_multiple(
            var_list,
            date,
            nsrdb_grid,
            nsrdb_freq=nsrdb_freq,
            dist_lim=dist_lim,
            var_meta=var_meta,
            max_workers=max_workers,
            max_workers_regrid=max_workers_regrid,
            fpath_out=fpath_out,
            scale=scale,
            factory_kwargs=factory_kwargs,
        )

        # Create an AncillaryDataProcessing object instance for storing data.
        logger.info(
            'Final NSRDB output shape is: {}'.format(
                data_model.nsrdb_data_shape
            )
        )

        logger.info('DataModel processing complete for: {}'.format(date))

        if return_obj:
            return data_model
        return data_model.processed_data

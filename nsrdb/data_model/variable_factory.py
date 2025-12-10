"""Factory pattern for retrieving NSRDB data source handlers."""

import logging
from typing import ClassVar

from nsrdb.data_model.albedo import AlbedoVar
from nsrdb.data_model.asymmetry import AsymVar
from nsrdb.data_model.base_handler import AncillaryVarHandler, BaseDerivedVar
from nsrdb.data_model.clouds import (
    CloudVar,
    CloudVarSingleH5,
    CloudVarSingleNC,
)
from nsrdb.data_model.gfs import GfsDewPoint, GfsVar
from nsrdb.data_model.maiac_aod import MaiacVar
from nsrdb.data_model.merra import DewPoint, MerraVar, RelativeHumidity
from nsrdb.data_model.nrel_data import NrelVar
from nsrdb.data_model.solar_zenith_angle import SolarZenithAngle

logger = logging.getLogger(__name__)


class VarFactory:
    """Factory pattern to retrieve ancillary variable helper objects."""

    # mapping of NSRDB variable names to helper objects
    MAPPING: ClassVar = {
        'asymmetry': AsymVar,
        'air_temperature': MerraVar,
        'surface_albedo': AlbedoVar,
        'alpha': MerraVar,
        'aod': MerraVar,
        'cloud_type': CloudVar,
        'cld_opd_dcomp': CloudVar,
        'cld_reff_dcomp': CloudVar,
        'cld_press_acha': CloudVar,
        'cloud_fraction': CloudVar,
        'cloud_probability': CloudVar,
        'temp_3_75um_nom': CloudVar,
        'temp_11_0um_nom': CloudVar,
        'temp_11_0um_nom_stddev_3x3': CloudVar,
        'refl_0_65um_nom': CloudVar,
        'refl_0_65um_nom_stddev_3x3': CloudVar,
        'refl_3_75um_nom': CloudVar,
        'dew_point': DewPoint,
        'ozone': MerraVar,
        'relative_humidity': RelativeHumidity,
        'solar_zenith_angle': SolarZenithAngle,
        'specific_humidity': MerraVar,
        'ssa': MerraVar,
        'surface_pressure': MerraVar,
        'total_precipitable_water': MerraVar,
        'wind_speed': MerraVar,
        'wind_direction': MerraVar,
        'wind_speed_10m': MerraVar,
        'wind_direction_10m': MerraVar,
    }

    HANDLER_NAMES: ClassVar = {
        'AsymVar': AsymVar,
        'AlbedoVar': AlbedoVar,
        'CloudVar': CloudVar,
        'GfsVar': GfsVar,
        'GfsDewPoint': GfsDewPoint,
        'MerraVar': MerraVar,
        'MaiacVar': MaiacVar,
        'NrelVar': NrelVar,
        'DewPoint': DewPoint,
        'RelativeHumidity': RelativeHumidity,
        'SolarZenithAngle': SolarZenithAngle,
    }

    @classmethod
    def get_instance(cls, var_name, *args, **kwargs):
        """Get a processing variable instance for the given var name.

        Parameters
        ----------
        var_name : str
            NSRDB variable name.
        *args : list
            List of positional args for instantiation of ancillary var.
        **kwargs : dict
            List of keyword args for instantiation of ancillary var.
            Can also include "handler" which specifies the handler name
            explicitly.

        Returns
        -------
        instance : ancillary object
            Instantiated ancillary variable helper object (AsymVar, MerraVar).
        """

        HandlerClass = cls.get_class(var_name, **kwargs)
        kwargs = cls._clean_kwargs(HandlerClass, kwargs)

        try:
            instance = HandlerClass(*args, **kwargs)
        except Exception as e:
            m = 'Received an exception trying to instantiate "{}":\n{}'.format(
                var_name, e
            )
            logger.exception(m)
            raise RuntimeError(m) from e

        return instance

    @classmethod
    def get_class(cls, var_name, **kwargs):
        """Get the Class that is used to process the target variable

        Parameters
        ----------
        var_name : str
            NSRDB variable name.
        **kwargs : dict
            List of keyword args for instantiation of ancillary var.
            Can also include "handler" which specifies the handler name
            explicitly.

        Returns
        -------
        HandlerClass : ancillary class
            Uninstantiated ancillary class (AsymVar, MerraVar).
        """

        if 'handler' in kwargs:
            handler = kwargs.pop('handler')
            if handler not in cls.HANDLER_NAMES:
                e = (
                    f'Did not recognize "{handler}" as an available NSRDB '
                    'variable data handler. The following handlers are '
                    f'available: {cls.HANDLER_NAMES.keys()}'
                )
                logger.error(e)
                raise KeyError(e)
            HandlerClass = cls.HANDLER_NAMES[handler]

        elif var_name in cls.MAPPING:
            HandlerClass = cls.MAPPING[var_name]

        else:
            e = (
                'Did not recognize "{}" as an available NSRDB '
                'variable. The following variables are available: '
                '{}'.format(var_name, list(cls.MAPPING.keys()))
            )
            logger.error(e)
            raise KeyError(e)

        return HandlerClass

    @classmethod
    def _clean_kwargs(
        cls,
        HandlerClass,
        kwargs,
        cld_list=('extent', 'cloud_dir', 'dsets', 'freq'),
    ):
        """Clean a kwargs namespace for cloud var specific kwargs.

        Parameters
        ----------
        HandlerClass : AncillaryVarHandler
            DataModel handler class. This method looks for the CloudVar class.
        kwargs : dict
            Namespace of kwargs to init HandlerClass.
        cld_list : tuple
            List of CloudVar specific input variables
            default: ('extent', 'cloud_dir', 'dsets')

        Returns
        -------
        kwargs : dict
            Namespace of kwargs to init handler class
            cleaned for cloud kwargs.
        """

        if 'handler' in kwargs:
            del kwargs['handler']

        if issubclass(HandlerClass, BaseDerivedVar):
            kwargs = {}

        elif HandlerClass == CloudVar:
            # handle legacy cloud dir kwarg
            if 'cloud_dir' in kwargs:
                kwargs['source_dir'] = kwargs.pop('cloud_dir')

        else:
            kwargs = {k: v for k, v in kwargs.items() if k not in cld_list}

        return kwargs

    @staticmethod
    def get_base_handler(*args, **kwargs):
        """Get the base ancillary variable handler to parse var meta.

        Parameters
        ----------
        *args : list
            List of positional args for instantiation of ancillary var.
        **kwargs : dict
            List of keyword args for instantiation of ancillary var.

        Returns
        -------
        instance : AncillaryVarHandler
            Instantiated base ancillary variable helper object.
        """

        return AncillaryVarHandler(*args, **kwargs)

    @classmethod
    def get_dset_attrs(cls, dset, var_meta=None):
        """Use the variable factory to get output attributes for a single dset.

        Parameters
        ----------
        dset : str
            Single dataset / variable name.
        var_meta : str | pd.DataFrame | None
            CSV file or dataframe containing meta data for all NSRDB variables.
            Defaults to the NSRDB var meta csv in git repo.

        Returns
        -------
        attrs : dict
            Dictionary of dataset attributes.
        chunks : tuple
            Chunk shape tuple.
        dtype : dict
            Numpy datatype.
        """

        var_obj = cls.get_base_handler(dset, var_meta=var_meta)

        return var_obj.attrs, var_obj.chunks, var_obj.final_dtype

    @classmethod
    def get_dsets_attrs(cls, dsets, var_meta=None):
        """Use the variable factory to get output attributes for list of dsets.

        Parameters
        ----------
        dsets : list
            List of dataset / variable names.
        var_meta : str | pd.DataFrame | None
            CSV file or dataframe containing meta data for all NSRDB variables.
            Defaults to the NSRDB var meta csv in git repo.

        Returns
        -------
        attrs : dict
            Dictionary of dataset attributes keyed by dset name.
        chunks : dict
            Dictionary of chunk tuples keyed by dset name.
        dtypes : dict
            dictionary of numpy datatypes keyed by dset name.
        """

        attrs = {}
        chunks = {}
        dtypes = {}

        for dset in dsets:
            out = cls.get_dset_attrs(dset, var_meta=var_meta)
            attrs[dset], chunks[dset], dtypes[dset] = out

        return attrs, chunks, dtypes

    @staticmethod
    def get_cloud_handler(fpath, dsets=None, **kwargs):
        """Get a cloud data file handler object.

        Parameters
        ----------
        fpath : str
            Full file path to cloud data file, must be either a .h5 or .nc
        dsets : NoneType | list | tuple
            Datasets of interest from cloud data file. None means only the
            grid data is of itnerest.

        Returns
        -------
        out : CloudVarSingleH5/NC
            Instantiated CloudVarSingle object appropraite for the .h5 or .nc
            fpath.
        """

        if kwargs:
            logger.debug(
                'Initializing cloud handler for datasets "{}" '
                'with kwargs: {} and fpath: {}'.format(dsets, kwargs, fpath)
            )

        kwarg_ignore = (
            'handler',
            'source_dir',
            'source_directory',
            'cloud_dir',
            'pattern',
        )
        kwargs = {k: v for k, v in kwargs.items() if k not in kwarg_ignore}

        if fpath.endswith('.h5'):
            return CloudVarSingleH5(fpath, dsets=dsets, **kwargs)
        if fpath.endswith('.nc'):
            return CloudVarSingleNC(fpath, dsets=dsets, **kwargs)
        raise TypeError(
            'Did not recognize cloud file type as .nc or ' '.h5: {}'.format(
                fpath
            )
        )

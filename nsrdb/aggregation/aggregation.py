# -*- coding: utf-8 -*-
"""NSRDB aggregation methods
- 2km 5min CONUS -> 4km 30min NSRDB PSM v3 Meta
- 2km 15min East -> 4km 30min NSRDB PSM v3 Meta
- 4km 30min West -> 4km 30min NSRDB PSM v3 Meta
"""

import logging
import os
import pickle
from concurrent.futures import as_completed
from typing import ClassVar
from warnings import warn

import numpy as np
import pandas as pd
from farms.utilities import calc_dhi
from rex import NSRDB as NSRDBHandler
from rex import MultiFileNSRDB
from rex.utilities.execution import SpawnProcessPool
from rex.utilities.loggers import init_logger
from scipy.spatial import cKDTree
from scipy.stats import mode

from nsrdb.file_handlers.outputs import Outputs
from nsrdb.utilities.interpolation import temporal_step
from nsrdb.utilities.plots import Spatial

logger = logging.getLogger(__name__)


# Standard configs.
NSRDB = {
    'full_disk': {
        'data_sub_dir': 'blended_full',
        'tree_file': 'kdtree_nsrdb_meta_2km.pkl',
        'meta_file': 'nsrdb_meta_2km.csv',
        'spatial': '2km',
        'freq': '10min',
    },
    'conus': {
        'data_sub_dir': 'blended_conus',
        'tree_file': 'kdtree_nsrdb_meta_2km_conus.pkl',
        'meta_file': 'nsrdb_meta_2km_conus.csv',
        'spatial': '2km',
        'freq': '5min',
    },
    'final': {
        'data_sub_dir': 'nsrdb_4km_30min',
        'fout': 'nsrdb.h5',
        'tree_file': 'kdtree_nsrdb_meta_4km.pkl',
        'meta_file': 'nsrdb_meta_4km.csv',
        'spatial': '4km',
        'freq': '30min',
        'source_priority': ['conus', 'full_disk'],
    },
}


SIMPLE = {
    'source': {
        'data_sub_dir': 'east',
        'tree_file': 'kdtree_nsrdb_meta_2km_east.pkl',
        'meta_file': 'nsrdb_meta_2km_east.csv',
        'spatial': '2km',
        'freq': '15min',
    },
    'final': {
        'data_sub_dir': 'wrf_9km',
        'fout': 'wrf_9km_2018.h5',
        'tree_file': 'kdtree_wrf_meta_9km.pkl',
        'meta_file': 'wrf_meta_9km.csv',
        'spatial': '9km',
        'freq': '15min',
    },
}


NSRDB_2018 = {
    'east': {
        'data_sub_dir': 'nsrdb_full_east_2018_2km_10min/final/',
        'tree_file': 'kdtree_nsrdb_meta_2km_full.pkl',
        'meta_file': 'nsrdb_meta_2km_full.csv',
        'spatial': '2km',
        'freq': '10min',
    },
    'conus': {
        'data_sub_dir': 'nsrdb_conus_east_2018_2km_5min/final/',
        'tree_file': 'kdtree_nsrdb_meta_2km_conus.pkl',
        'meta_file': 'nsrdb_meta_2km_conus.csv',
        'spatial': '2km',
        'freq': '5min',
    },
    'west': {
        'data_sub_dir': 'nsrdb_west_full_2018_4km_30min/final/',
        'tree_file': 'kdtree_nsrdb_meta_4km.pkl',
        'meta_file': 'nsrdb_meta_4km.csv',
        'spatial': '4km',
        'freq': '30min',
    },
    'final': {
        'data_sub_dir': 'nsrdb_4km_30min',
        'fout': 'nsrdb_2018.h5',
        'tree_file': 'kdtree_nsrdb_meta_4km.pkl',
        'meta_file': 'nsrdb_meta_4km.csv',
        'spatial': '4km',
        'freq': '30min',
    },
}


SURFRAD_2018 = {
    'east': {
        'data_sub_dir': 'east',
        'tree_file': 'kdtree_nsrdb_meta_2km_east.pkl',
        'meta_file': 'nsrdb_meta_2km_east.csv',
        'spatial': '2km',
        'freq': '15min',
    },
    'west': {
        'data_sub_dir': 'west',
        'tree_file': 'kdtree_west_psm_extent.pkl',
        'meta_file': 'west_psm_extent.csv',
        'spatial': '4km',
        'freq': '30min',
    },
    'conus': {
        'data_sub_dir': 'conus',
        'tree_file': 'kdtree_nsrdb_meta_2km_conus.pkl',
        'meta_file': 'nsrdb_meta_2km_conus.csv',
        'spatial': '2km',
        'freq': '5min',
    },
    'final': {
        'data_sub_dir': 'nsrdb_4km_30min',
        'fout': 'nsrdb_surfrad_2018.h5',
        'tree_file': 'kdtree_surfrad_meta.pkl',
        'meta_file': 'surfrad_meta.csv',
        'spatial': '4km',
        'freq': '30min',
    },
}


class MetaManager:
    """Framework to parse the final meta data for contributing sources."""

    @staticmethod
    def meta_sources(fpath_final, source_tree_fpaths, source_priority=None):
        """Make a final meta data with data source columns (full_disk/conus).

        Parameters
        ----------
        fpath_final : str
            File path to final output grid (usually full 4km meta data).
        source_tree_fpaths : dict
            Dictionary mapping the source name to pickled meta kdtree
            pickle filepaths.
        source_priority : list
            Priority list (high to low) for meta sources.

        Returns
        -------
        meta : pd.DataFrame
            DataFrame based on fpath_4km but with a "source" column containing
            the data source string.
        """
        logger.info(
            f'Making aggregation meta data from {len(source_tree_fpaths)} '
            f'sources to final meta: {fpath_final}. Aggregation meta data '
            f'sources: {source_tree_fpaths}'
        )

        meta = pd.read_csv(fpath_final, index_col=0)
        meta['source'] = 'null'

        source_iter = source_tree_fpaths.keys()
        if source_priority is not None:
            source_iter = source_priority

        d_last = np.ones(len(meta)) * 1e6
        for source_name in source_iter:
            tree_fpath = source_tree_fpaths[source_name]
            tree = MetaManager.load_pickle_tree(tree_fpath)
            d = tree.query(meta[['latitude', 'longitude']].values)[0]
            mask = d < d_last
            logger.debug(
                'Source "{}" found {} locations in final meta'.format(
                    source_name, mask.sum()
                )
            )
            ind = meta.index[mask]
            meta.loc[ind, 'source'] = source_name
            d_last = d

        for name in pd.unique(meta['source']):
            count = (meta['source'] == name).sum()
            logger.info(
                f'Aggregated NSRDB meta data has {count} sites with '
                f'aggregation source "{name}"'
            )

        if any(meta['source'] == 'null'):
            e = 'Null source values persisted in aggregation meta'
            logger.error(e)
            raise RuntimeError(e)

        return meta

    @staticmethod
    def meta_sources_2018(fpath_4km):
        """Make 2018 4km meta data with data source column (west/east/conus).

        WARNING: This is a script very specific to the 2018 GOES data
        arrangement, with the 4km 30min GOES West satellite, the 2km 15min GOES
        East satellite, and the 2km 5min CONUS data from GOES East. This only
        works with the psm v3 4km meta data (accessed on 8/28/2019).

        Parameters
        ----------
        fpath_4km : str
            File path to full 4km meta data.

        Returns
        -------
        meta : pd.DataFrame
            DataFrame based on fpath_4km but with a "source" column containing
            the data source string.
        """
        logger.info(
            'Making meta data for custom 2018 setup with 4km '
            'source file: {}'.format(fpath_4km)
        )

        meta = pd.read_csv(fpath_4km, index_col=0)
        meta['source'] = 'west'

        # east 2km longitude boundary is at -125 lon (just west of CONUS)
        east_mask = meta.longitude > -125.0
        meta.loc[east_mask, 'source'] = 'east'

        # conus includes all of US except for Alaska and Hawaii
        conus_mask = (meta.country == 'United States') & ~meta.state.isin([
            'Alaska',
            'Hawaii',
        ])
        meta.loc[conus_mask, 'source'] = 'conus'

        # made a line specific to the observed 2018 GOES East extreme angle
        # boundary, above which no cloud properties are returned for the East.
        lat_boundary = 0.6 * (meta.longitude.values + 125) + 42.7
        angle_mask = (
            (meta.latitude > lat_boundary)
            & (meta.source != 'west')
            & (meta.longitude < -104.5)
        )
        meta.loc[angle_mask, 'source'] = 'west'

        return meta

    @staticmethod
    def simple(fpath_meta):
        """Make a meta dataframe with source column == 'source' for simple agg

        Parameters
        ----------
        fpath_meta : str
            File path to final meta data.

        Returns
        -------
        meta : pd.DataFrame
            DataFrame based on fpath_meta but with a "source" column containing
            the data source string 'source'.
        """

        logger.info(
            'Making simple meta data with final meta: {}'.format(fpath_meta)
        )
        meta = pd.read_csv(fpath_meta, index_col=0)
        meta['source'] = 'source'
        return meta

    @staticmethod
    def load_pickle_tree(fpath):
        """Load a pickled ckdtree and raise exception if not loadable

        Parameters
        ----------
        fpath : str
            Full filepath to pickle file.

        Returns
        -------
        tree : ckdtree
            Un-pickled ckdtree to query.
        """

        if not os.path.exists(fpath):
            e = 'Could not load pickle file, does not exist: {}'.format(fpath)
            logger.error(e)
            raise FileNotFoundError(e)

        try:
            with open(fpath, 'rb') as pkl:
                tree = pickle.load(pkl)
        except Exception as e:
            m = (
                'Could not load pickle file. May have been generated with a '
                f'different python version. Try recreating the file: {e}'
            )
            logger.error(m)
            raise e

        return tree

    @staticmethod
    def plot_meta_source(fpath_meta, fname, out_dir, **kwargs):
        """Make a map plot of the NSRDB Meta source data (west/east/conus).

        Parameters
        ----------
        fpath_meta : str
            File path to aggregation meta with source column
        fname : str
            Filename for output map image file.
        out_dir : str
            Directory path to save map plot file.
        **kwargs : dict
            Keyword args for spatial plotting utility.
        """

        meta = pd.read_csv(fpath_meta)
        sources = list(set(meta.source.unique()))
        meta['isource'] = np.nan
        for i, source in enumerate(sources):
            meta.loc[(meta.source == source), 'isource'] = i

        meta = meta[['latitude', 'longitude', 'isource']]
        Spatial.plot_geo_df(meta, fname, out_dir, **kwargs)


class Aggregation:
    """Framework for performing spatiotemporal aggregation."""

    def __init__(self, var, data_fpath, nn, w, final_ti):
        """
        Parameters
        ----------
        var : str
            Variable (dataset) name being aggregated.
        data_fpath : str
            Filepath to h5 file containing source var data.
        nn : np.ndarray
            1D array of site (column) indices in data_fpath to aggregate.
        w : int
            Window size for temporal aggregation.
        final_ti : pd.DateTimeIndex
            Final datetime index (used to ensure the aggregated profile has
            correct length).
        """
        self.var = var
        self.data_fpath = data_fpath
        self.nn = sorted(nn)
        self.w = w
        self.final_ti = final_ti

    @property
    def source_time_index(self):
        """Get the time index of the source data.

        Returns
        -------
        time_index : pd.Datetimeindex
            Datetimeindex of the source dataset.
        """

        if os.path.isfile(self.data_fpath):
            with NSRDBHandler(self.data_fpath) as f:
                ti = f.time_index
        else:
            with MultiFileNSRDB(self.data_fpath) as f:
                ti = f.time_index
        return ti

    @property
    def data(self):
        """Get the timeseries data for the specified var and sites.

        Returns
        -------
        _data : np.ndarray
            Unscaled float data array with shape (ti, nn) where ti is the
            native time index length and nn is the number of neighbors in
            the self.nn attr.
        """

        if os.path.isfile(self.data_fpath):
            with NSRDBHandler(self.data_fpath, unscale=False) as f:
                _data = f[self.var, :, self.nn].astype(np.float32)
        else:
            with MultiFileNSRDB(self.data_fpath, unscale=False) as f:
                _data = f[self.var, :, self.nn].astype(np.float32)
        return _data

    @staticmethod
    def spatial_avg(data):
        """Average the source data across the spatial extent.

        Returns
        -------
        data : np.ndarray
            Unscaled float data array with shape (ti, ) where ti is the
            native time index length the data was averaged accross all
            nn neighbors.
        """

        bad_rows = np.isnan(data).all(axis=1)

        if any(bad_rows):
            data[bad_rows, :] = 0
            m = np.nanmean(data, axis=1)
            m[bad_rows] = np.nan
        else:
            m = np.mean(data, axis=1)

        return m

    @staticmethod
    def spatial_sum(data):
        """Sum the source data across the spatial extent.

        Returns
        -------
        data : np.ndarray
            Unscaled float data array with shape (ti, ) where ti is the
            native time index length the data was summed accross all
            nn neighbors.
        """

        return data.sum(axis=1)

    def time_avg(self, inp):
        """Calculate the rolling time average for an input array or df.

        Parameters
        ----------
        inp : np.ndarray | pd.DataFrame
            Input array/df with data to average.

        Returns
        -------
        out : np.ndarray | pd.DataFrame
            Array or dataframe with same size as input and each value is a
            moving average.
        """

        if self.w > 1:
            array = False
            if isinstance(inp, np.ndarray):
                array = True
                inp = pd.DataFrame(inp)

            out = inp.rolling(self.w, center=True, min_periods=1).mean()

            if array:
                out = out.values

            return out

        return inp

    def time_sum(self, inp):
        """Calculate the rolling sum for an input array or df.

        Parameters
        ----------
        inp : np.ndarray | pd.DataFrame
            Input array/df with data to sum.

        Returns
        -------
        out : np.ndarray | pd.DataFrame
            Array or dataframe with same size as input and each value is a
            moving sum.
        """

        if self.w > 1:
            array = False
            if isinstance(inp, np.ndarray):
                array = True
                inp = pd.DataFrame(inp)

            out = inp.rolling(self.w, center=True, min_periods=1).sum()

            if array:
                out = out.values

            return out

        return inp

    @staticmethod
    def _get_rolling_window_index(L, w):
        """Get an indexing array to index a 2D array on a rolling window.

        Parameters
        ----------
        L : int
            Length of the 2D array to apply a rolling window over
        w : int
            Window size.

        Returns
        -------
        iarr : np.ndarray
            Array of index arrays.
        """
        if w == 1:
            iarr = np.expand_dims(np.arange(L), axis=1)

        else:
            iarr = np.zeros((L, w), dtype=np.int32)
            for i in range(L):
                sub = [i + n for n in range(w)]
                sub -= np.round(w / 2) - 1
                iarr[i, :] = sub
            iarr[(iarr < 0)] = 0
            iarr = np.maximum(iarr, 0)
            iarr = np.minimum(iarr, L - 1)

        return iarr

    @staticmethod
    def cloud_type_mode(data, w):
        """Get the mode of a 2D cloud type array using a rolling time window.

        Parameters
        ----------
        data : np.ndarray
            2D array of integer cloud types.
        w : int
            Temporal window over which to take the mode.

        Returns
        -------
        data : np.ndarray
            Mode of cloud type.
        """

        iarr = Aggregation._get_rolling_window_index(len(data), w)

        cloud_mode = np.ndarray((len(data),), dtype=data.dtype)
        for i, x in enumerate(data[iarr]):
            m = mode(x.flatten())[0]
            if np.isnan(m):
                emsg = 'Bad cloud type mode from ctype array: \n\t{}'.format(x)
                logger.info('Bad ctype mode at i {}'.format(i))
                logger.exception(emsg)
                raise ValueError(emsg)
            cloud_mode[i] = m

        return cloud_mode

    def reduce_timeseries(self, arr):
        """Reduce a high res timeseries to a coarse timeseries.

        Parameters
        ----------
        arr : np.ndarray
            2D numpy array

        Returns
        -------
        arr : np.ndarray
            Shortened 2D numpy array with length equal to the final ti.
        """

        m = len(arr) / len(self.final_ti)
        if m % 1 != 0:
            raise ValueError(
                'Cannot reduce timeseries! Final ti has shape '
                '{} and working array has shape {}.'.format(
                    self.final_ti.shape, arr.shape
                )
            )
        m = int(m)

        arr = arr[::m] if len(arr.shape) == 1 else arr[::m, :]

        if len(arr) != len(self.final_ti):
            raise ValueError(
                'Timeseries length reduction failed! Final ti '
                'has shape {} and reduced array has shape {}.'.format(
                    self.final_ti.shape, arr.shape
                )
            )

        return arr

    @staticmethod
    def format_out_arr(arr):
        """Format the output array (round and flatten)."""
        return np.round(arr).flatten()

    @classmethod
    def point(cls, var, data_fpath, nn, w, final_ti):
        """Run agg by selecting just the closest site and timestep.

        Parameters
        ----------
        var : str
            Variable (dataset) name being aggregated.
        data_fpath : str
            Filepath to h5 file containing source var data.
        nn : np.ndarray
            1D array of site (column) indices in data_fpath to aggregate.
        w : int
            Window size for temporal aggregation.
        final_ti : pd.DateTimeIndex
            Final datetime index (used to ensure the aggregated profile has
            correct length).

        Returns
        -------
        data : np.ndarray
            (n, ) array unscaled and rounded data from the nn with time
            series matching final_ti.
        """
        nn = [next(iter(nn))]
        a = cls(var, data_fpath, nn, w, final_ti)
        data = a.reduce_timeseries(a.data)
        data = a.format_out_arr(data)
        return data

    @classmethod
    def dhi(cls, var, i, fout):
        """Calculate the aggregated DHI from an aggregated output file.

        Parameters
        ----------
        var : str
            Variable name, either "dhi" or "clearsky_dhi".
        i : int
            Site index in fout.
        fout : str
            Filepath to the output file containing aggregated GHI, DNI, and
            SZA to calculate aggregated DHI.

        Returns
        -------
        dhi : np.ndarray
            DHI calcualted from vars in fout.
        """

        var_ghi = var.replace('dhi', 'ghi')
        var_dni = var.replace('dhi', 'dni')

        with NSRDBHandler(fout) as out:
            attrs = out.get_attrs(dset=var)
            ghi = out[var_ghi, :, i]
            dni = out[var_dni, :, i]
            sza = out['solar_zenith_angle', :, i]

        dhi = calc_dhi(dni, ghi, sza)[0]
        dhi *= float(attrs.get('scale_factor', 1))
        dhi = cls.format_out_arr(dhi)
        return dhi

    @classmethod
    def fill_flag(cls, var, data_fpath, nn, w, final_ti):
        """Run fill flag aggregation, returning the percentage of timesteps
        that were filled.

        Parameters
        ----------
        var : str
            Variable (dataset) name being aggregated (fill_flag).
        data_fpath : str
            Filepath to h5 file containing source var data.
        nn : np.ndarray
            1D array of site (column) indices in data_fpath to aggregate.
        w : int
            Window size for temporal aggregation.
        final_ti : pd.DateTimeIndex
            Final datetime index (used to ensure the aggregated profile has
            correct length).

        Returns
        -------
        data : np.ndarray
            (n, ) array unscaled and rounded data from the nn with time
            series matching final_ti.
        """
        a = cls(var, data_fpath, nn, w, final_ti)
        data = a.data
        data[(data > 0)] = 1
        data = a.spatial_sum(data)
        data = a.time_sum(data)
        data = a.reduce_timeseries(data)
        data /= len(nn) * w / 100
        data = a.format_out_arr(data)
        return data

    @classmethod
    def cloud_type(cls, var, data_fpath, nn, w, final_ti):
        """Run cloud type aggregation, returning the most common cloud type.

        Parameters
        ----------
        var : str
            Variable (dataset) name being aggregated (cloud_type).
        data_fpath : str
            Filepath to h5 file containing source var data.
        nn : np.ndarray
            1D array of site (column) indices in data_fpath to aggregate.
        w : int
            Window size for temporal aggregation.
        final_ti : pd.DateTimeIndex
            Final datetime index (used to ensure the aggregated profile has
            correct length).

        Returns
        -------
        data : np.ndarray
            (n, ) array unscaled and rounded data from the nn with time
            series matching final_ti.
        """
        a = cls(var, data_fpath, nn, w, final_ti)
        data = a.data
        data[(data == 1)] = 0
        data = a.cloud_type_mode(data, w)
        data = a.reduce_timeseries(data)
        data = a.format_out_arr(data)
        return data

    @staticmethod
    def cloud_property_avg(cprop_source, ctype_source, ctype_out_full, w):
        """Run cloud property aggregation based on output cloud type.

        Parameters
        ----------
        cprop_source : np.ndarray
            Source (full resolution) cloud property data.
        ctype_source : np.ndarray
            Source (full resolution) cloud type data.
        ctype_out_full : np.ndarray
            Output (reduced resolution) cloud type data, interpolated to the
            same length as the source resolution.
        w : int
            Window size.

        Returns
        -------
        cprop_out : np.ndarray
            Average cloud property data in the window surrounding each
            timestep masked by cloud type output == cloud type source.
            Shape is same as ctype_out_full.
        """

        iarr = Aggregation._get_rolling_window_index(len(ctype_source), w)

        cprop_out = np.ndarray((len(cprop_source),), dtype=cprop_source.dtype)

        for i, j in enumerate(iarr):
            mask = ctype_source[j] == ctype_out_full[i]
            if mask.any():
                cprop_out[i] = np.mean(cprop_source[j][mask])
            else:
                cprop_out[i] = np.nan

        cprop_out[(ctype_out_full[:, 0] == 0)] = 0

        return cprop_out

    @classmethod
    def cloud_property(cls, var, data_fpath, nn, w, final_ti, gid, fout):
        """Run cloud property aggregation, returning the mean cloud property
        only for timesteps that match the most common (mode) cloud type.

        Parameters
        ----------
        var : str
            Variable (dataset) name being aggregated (cloud_type).
        data_fpath : str
            Filepath to h5 file containing source var data.
        nn : np.ndarray
            1D array of site (column) indices in data_fpath to aggregate.
        w : int
            Window size for temporal aggregation.
        final_ti : pd.DateTimeIndex
            Final datetime index (used to ensure the aggregated profile has
            correct length).
        gid : int
            Site index in fout.
        fout : str
            Filepath to the output file containing aggregated cloud type.

        Returns
        -------
        data : np.ndarray
            Average cloud property data in the window surrounding each
            timestep masked by cloud type output == cloud type source.
            Shape is same as ctype_out_full. Array is (n, ) and is unscaled
            and rounded data from the nn with time series matching final_ti.
        """

        with NSRDBHandler(fout) as out:
            i = np.where(out.meta.gid == gid)[0][0]
            ctype_out_final = out['cloud_type', :, i]
            sza = out['solar_zenith_angle', :, i]

        a = cls('cloud_type', data_fpath, nn, w, final_ti)
        ctype_source = a.data
        ctype_source[(ctype_source == 1)] = 0

        ctype_out_full = temporal_step(
            ctype_out_final, final_ti, a.source_time_index
        )

        a = cls(var, data_fpath, nn, w, final_ti)
        cprop_source = a.data

        cprop_out = a.cloud_property_avg(
            cprop_source, ctype_source, ctype_out_full, w
        )

        cprop_out = a.reduce_timeseries(cprop_out)
        cprop_out = a.format_out_arr(cprop_out)

        if np.isnan(cprop_out).sum():
            emsg = (
                'Aggregation of cloud property failed for site gid '
                '{}, {} NaN values persisted.'.format(
                    gid, np.isnan(cprop_out).sum()
                )
            )
            bad_locs = np.where(np.isnan(cprop_out))[0]
            logger.info('Bad locs: {}'.format(bad_locs))
            logger.info('Bad time index: {}'.format(final_ti[bad_locs]))
            logger.info('Ctypes: {}'.format(ctype_out_final[bad_locs]))
            logger.info('SZA: {}'.format(sza[bad_locs]))
            i0 = np.max(bad_locs[0], 0) - 5
            i1 = i0 + 10
            logger.info(
                'Cprop from {} to {}: {}'.format(i0, i1, cprop_out[i0:i1])
            )
            logger.exception(emsg)
            raise ValueError(emsg)

        return cprop_out

    @classmethod
    def mean(cls, var, data_fpath, nn, w, final_ti):
        """Run agg using a spatial average and temporal moving window average.

        Parameters
        ----------
        var : str
            Variable (dataset) name being aggregated.
        data_fpath : str
            Filepath to h5 file containing source var data.
        nn : np.ndarray
            1D array of site (column) indices in data_fpath to aggregate.
        w : int
            Window size for temporal aggregation.
        final_ti : pd.DateTimeIndex
            Final datetime index (used to ensure the aggregated profile has
            correct length).

        Returns
        -------
        data : np.ndarray
            (n, ) array unscaled and rounded data from the nn with time
            series matching final_ti.
        """

        a = cls(var, data_fpath, nn, w, final_ti)
        data = a.spatial_avg(a.data)
        data = a.time_avg(data)
        data = a.reduce_timeseries(data)
        data = a.format_out_arr(data)
        return data


class Manager:
    """Framework for aggregation to a final NSRDB spatiotemporal resolution."""

    DEFAULT_METHOD = Aggregation.mean

    AGG_METHODS: ClassVar = {
        'alpha': Aggregation.point,
        'aod': Aggregation.point,
        'asymmetry': Aggregation.point,
        'ozone': Aggregation.point,
        'ssa': Aggregation.point,
        'surface_albedo': Aggregation.point,
        'surface_pressure': Aggregation.point,
        'total_precipitable_water': Aggregation.point,
        'dew_point': Aggregation.point,
        'relative_humidity': Aggregation.point,
        'air_temperature': Aggregation.point,
        'wind_direction': Aggregation.point,
        'wind_speed': Aggregation.point,
        'wind_direction_10m': Aggregation.point,
        'wind_speed_10m': Aggregation.point,
        'cloud_type': Aggregation.cloud_type,
        'cld_opd_dcomp': Aggregation.cloud_property,
        'cld_reff_dcomp': Aggregation.cloud_property,
        'cld_press_acha': Aggregation.cloud_property,
        'solar_zenith_angle': Aggregation.point,
        'dhi': Aggregation.dhi,
        'dni': Aggregation.mean,
        'ghi': Aggregation.mean,
        'clearsky_dhi': Aggregation.dhi,
        'clearsky_dni': Aggregation.mean,
        'clearsky_ghi': Aggregation.mean,
        'fill_flag': Aggregation.fill_flag,
    }

    def __init__(
        self, data, data_dir, meta_dir, year=2018, i_chunk=0, n_chunks=1
    ):
        """
        Parameters
        ----------
        data : dict
            Nested dictionary containing data on all NSRDB data sources
            (east, west, conus) and the final aggregated output.
        data_dir : str
            Root directory containing sub dirs with all data sources.
        meta_dir : str
            Directory containing meta and ckdtree files for each data source
            and the final aggregated output.
        year : int
            Year being analyzed.
        i_chunk : int
            Meta data chunk index currently being processed (zero indexed).
        n_chunks : int
            Number of chunks to process the meta data in.
        """

        self.data = data
        self.data_dir = data_dir
        self.meta_dir = meta_dir
        self.year = year
        self.n_chunks = n_chunks
        self.i_chunk = i_chunk
        self._meta = None
        self._meta_chunk = None

        self.final_sres = None
        self.final_tres = None
        self.fout = None
        self.data_sources = None
        self.dsets = None
        self.attrs = None
        self.chunks = None
        self.dtypes = None

        self.parse_data()
        self.preflight()
        self.run_nn()
        self.add_temporal()
        self._init_fout()

    def parse_data(self):
        """Parse the data input for several useful attributes."""
        self.final_sres = self.data['final']['spatial']
        self.final_tres = self.data['final']['freq']
        if 'fpath' in self.data['final']:
            self.fout = self.data['final']['fpath']
        else:
            self.fout = os.path.join(
                self.data_dir,
                self.data['final']['data_sub_dir'] + '/',
                self.data['final']['fout'],
            )
        self.fout = self.fout.replace('.h5', '_{}.h5'.format(self.i_chunk))
        self.data_sources = self.meta.source.unique()

        logger.info('Data sources: {}'.format(self.data_sources))

        out_dir = os.path.dirname(self.fout)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

    def preflight(
        self,
        reqs=('data_sub_dir', 'tree_file', 'meta_file', 'spatial', 'freq'),
    ):
        """Run validity checks on input data.

        Parameters
        ----------
        reqs : list | tuple
            Required fields for each source dataset.
        """

        sources = self.data_sources
        for source in sources:
            if source not in self.data:
                warn('Need "{}" in the data input!'.format(source))
            for r in reqs:
                if r not in self.data[source]:
                    warn(
                        'Data input source "{}" needs field "{}"!'.format(
                            source, r
                        )
                    )

    @property
    def time_index(self):
        """Get the final time index.

        Returns
        -------
        ti : pd.DatetimeIndex
            Time index for the intended year at the final (aggregated) time
            resolution.
        """

        ti = pd.date_range(
            '1-1-{}'.format(self.year),
            '1-1-{}'.format(self.year + 1),
            freq=self.final_tres,
        )[:-1]
        return ti

    @property
    def meta(self):
        """Get the final meta data with sources.

        Returns
        -------
        meta : pd.DataFrame
            Meta data for the final (aggregated) datasets with data source col.
        """

        if self._meta is None:
            final_meta_path = os.path.join(
                self.meta_dir, self.data['final']['meta_file']
            )

            source_tree_fpaths = {}
            for source_name, attrs in self.data.items():
                if source_name.lower() != 'final':
                    p = os.path.join(self.meta_dir, attrs['tree_file'])
                    source_tree_fpaths[source_name] = p

            if 'nsrdb_2018' in os.path.basename(self.fout):
                self._meta = MetaManager.meta_sources_2018(final_meta_path)

            elif len(source_tree_fpaths) == 1:
                self._meta = MetaManager.simple(final_meta_path)

            else:
                priority = self.data['final'].get('source_priority', None)
                self._meta = MetaManager.meta_sources(
                    final_meta_path,
                    source_tree_fpaths,
                    source_priority=priority,
                )

        return self._meta

    @property
    def meta_chunk(self):
        """Get the meta data for just this chunk of sites based on
        n_chunks and i_chunk.

        Returns
        -------
        meta_chunk : pd.DataFrame
            Meta data reduced to a chunk of sites based on n_chunks and i_chunk
        """

        if self._meta_chunk is None:
            gids_full = np.arange(len(self.meta))
            gid_chunk = np.array_split(gids_full, self.n_chunks)[self.i_chunk]
            self._meta_chunk = self._meta.iloc[gid_chunk, :]
            self._meta_chunk['gid'] = self._meta_chunk.index.values
            logger.info(
                'Working on meta chunk {} out of {} '
                'with GIDs {} through {}'.format(
                    self.i_chunk + 1,
                    self.n_chunks,
                    np.min(gid_chunk),
                    np.max(gid_chunk),
                )
            )
        return self._meta_chunk

    @staticmethod
    def _special_attrs(dset, dset_attrs):
        """Enforce any special dataset attributes.

        Parameters
        ----------
        dset : str
            Name of dataset
        dset_attrs : dict
            Attribute key-value pair dictionary for dset.

        Returns
        -------
        dset_attrs : dict
            Attributes for dset with any special formatting.
        """

        if 'fill_flag' in dset:
            dset_attrs['units'] = 'percent of filled timesteps'

        if (
            'scale_factor' in dset_attrs
            and 'psm_scale_factor' not in dset_attrs
        ):
            dset_attrs['psm_scale_factor'] = dset_attrs['scale_factor']

        if 'units' in dset_attrs and 'psm_units' not in dset_attrs:
            dset_attrs['psm_units'] = dset_attrs['units']

        return dset_attrs

    @staticmethod
    def get_dset_attrs(
        h5dir, ignore_dsets=('coordinates', 'time_index', 'meta')
    ):
        """Get output file dataset attributes for a set of datasets.

        Parameters
        ----------
        h5dir : str
            Path to directory containing multiple h5 files with all available
            dsets. Can also be a single h5 filepath.
        ignore_dsets : tuple | list
            List of datasets to ignore (will not be aggregated).

        Returns
        -------
        dsets : list
            List of datasets.
        attrs : dict
            Dictionary of dataset attributes keyed by dset name.
        chunks : dict
            Dictionary of chunk tuples keyed by dset name.
        dtypes : dict
            dictionary of numpy datatypes keyed by dset name.
        ti : pd.Datetimeindex
            Time index of source files in h5dir.
        """

        dsets = []
        attrs = {}
        chunks = {}
        dtypes = {}

        if h5dir.endswith('.h5') and os.path.isfile(h5dir):
            h5_files = [h5dir]
        elif h5dir.endswith('.h5') and '*' in h5dir:
            with MultiFileNSRDB(h5dir) as res:
                h5_files = res._h5_files
        else:
            h5_files = [fn for fn in os.listdir(h5dir) if fn.endswith('.h5')]

        logger.info(
            'Getting dataset attributes from the following files: {}'.format(
                h5_files
            )
        )

        for fn in h5_files:
            with NSRDBHandler(os.path.join(h5dir, fn)) as out:
                ti = out.time_index
                for d in out.dsets:
                    if d not in ignore_dsets and d not in attrs:
                        attrs[d] = Manager._special_attrs(
                            d, out.get_attrs(dset=d)
                        )

                        try:
                            x = out.get_dset_properties(d)
                        except Exception as e:
                            m = (
                                'Could not get dataset "{}" properties from '
                                'file: {}'.format(d, os.path.join(h5dir, fn))
                            )
                            logger.error(m)
                            logger.exception(m)
                            raise e
                        else:
                            _, dtypes[d], chunks[d] = x

        dsets = list(attrs.keys())
        logger.info('Found the following datasets: {}'.format(dsets))

        return dsets, attrs, chunks, dtypes, ti

    def _init_fout(self):
        """Initialize the output file with all datasets and final time index
        and meta"""

        if 'fpath' in self.data[self.data_sources[0]]:
            self.dsets, self.attrs, self.chunks, self.dtypes, _ = (
                self.get_dset_attrs(self.data[self.data_sources[0]]['fpath'])
            )
        else:
            data_sub_dir = self.data[self.data_sources[0]]['data_sub_dir']
            data_sub_dir += '/'
            self.dsets, self.attrs, self.chunks, self.dtypes, _ = (
                self.get_dset_attrs(os.path.join(self.data_dir, data_sub_dir))
            )

        if not os.path.exists(self.fout):
            logger.info('Initializing output file: {}'.format(self.fout))
            Outputs.init_h5(
                self.fout,
                self.dsets,
                self.attrs,
                self.chunks,
                self.dtypes,
                self.time_index,
                self.meta_chunk,
            )
        else:
            logger.info('Output file exists: {}'.format(self.fout))

    def _init_arr(self, var):
        """Initialize a numpy array for a given var for current chunk of sites.

        Returns
        -------
        arr : np.ndarray
            Numpy array for var with final disk dtype and shape (t, n) where
            t is the final time index and n is the number of sites in this
            meta chunk.
        """
        arr = np.zeros(
            (len(self.time_index), len(self.meta_chunk)), self.dtypes[var]
        )
        logger.debug(
            'Initializing array for "{}" with shape {}'.format(var, arr.shape)
        )
        return arr

    @staticmethod
    def _get_spatial_k(sres, final_sres):
        """Get the required number of nearest neighbors based on spatial res.

        Parameters
        ----------
        sres : str
            Spatial resolution of the source data (4km, 2km).
        final_sres : str
            Spatial resolution of the final dataset (4km).

        Returns
        -------
        k : int
            Number of neighbors in the source data to aggregate to a final
            dataset site.
        """
        if final_sres == sres:
            k = 1

        else:
            try:
                s1 = int(sres.replace('km', ''))
                s2 = int(final_sres.replace('km', ''))
                k = (s2**2) / (s1**2)
            except Exception as e:
                msg = (
                    'Could not parse spatial resolution from: '
                    '"{}", "{}"'.format(sres, final_sres)
                )
                logger.error(msg)
                raise ValueError(msg) from e
            else:
                k = int(np.ceil(k))

        return k

    @staticmethod
    def _get_temporal_w(tres, final_tres):
        """Get the required moving window size for temporal agg

        Parameters
        ----------
        tres : str
            Temporal resolution of the source data (30min, 15min, 5min).
        final_tres : str
            Temporal resolution of the final dataset (30min).

        Returns
        -------
        w : int
            Window size to consider for the temporal aggregation to the
            final_tres.
        """
        if final_tres == tres:
            w = 1

        elif final_tres == '30min':
            if tres == '15min':
                w = 3
            elif tres == '10min':
                w = 4
            elif tres == '5min':
                w = 7
            else:
                raise ValueError(
                    'Did not recognize temporal resolution {}'.format(tres)
                )
        else:
            raise ValueError(
                'Did not recognize final temporal resolution: ' '{}'.format(
                    final_tres
                )
            )

        return w

    def _get_fpath(self, var, data_dir, data_sub_dir, source):
        """Get the h5 filepath in data_dir/data_sub_dir/ containing var.

        Parameters
        ----------
        var : str
            Variable name (h5 dataset) being searched for.
        data_dir : str
            Root data directory.
        data_sub_dir : str
            Sub directory in data_dir containing h5 files.
        source : str
            Data source (conus, east, west).

        Returns
        -------
        fpath : str
            File path to h5 file in data_sub_dir containing var dataset.
        """

        if var in self.data[source]:
            fpath = self.data[source][var]
        else:
            if not data_sub_dir.endswith('/'):
                data_sub_dir += '/'
            for fn in os.listdir(os.path.join(data_dir, data_sub_dir)):
                if fn.endswith('.h5'):
                    fpath = os.path.join(data_dir, data_sub_dir, fn)
                    with NSRDBHandler(fpath) as out:
                        if (
                            var == 'fill_flag'
                            and var in out.dsets
                            and 'irradiance' in fn
                        ) or (var != 'fill_flag' and var in out.dsets):
                            break
            self.data[source][var] = fpath
        return fpath

    def add_temporal(self):
        """Get the temporal window sizes for all data sources."""
        for source in self.data_sources:
            w = self._get_temporal_w(
                self.data[source]['freq'], self.final_tres
            )
            self.data[source]['window'] = w
            logger.info(
                'Data source {} has a window size of {}'.format(source, w)
            )

    def run_nn(self):
        """Run nearest neighbor for all data sources against the final meta."""
        for source in self.data_sources:
            k = self._get_spatial_k(
                self.data[source]['spatial'], self.final_sres
            )
            logger.info(
                'Data source {} will use {} neighbors'.format(source, k)
            )

            meta_fpath = os.path.join(
                self.meta_dir, self.data[source]['meta_file']
            )
            tree_fpath = os.path.join(
                self.meta_dir, self.data[source]['tree_file']
            )
            _, i = self.knn(self.meta, tree_fpath, meta_fpath, k=k)

            self.data[source]['nn'] = i

    @staticmethod
    def knn(meta, tree_fpath, meta_fpath, k=1):
        """Run KNN between the final meta data and the pickled ckdtree.

        Parameters
        ----------
        meta : pd.DataFrame
            Final meta data.
        tree_fpath : str
            Filepath to a pickled ckdtree containing ckdtree for source
            meta data.
        meta_fpath : str
            Filepath to csv containing source meta data.
        k : int
            Number of neighbors to query.

        Returns
        -------
        d : np.ndarray
            Distance results. Shape is (len(meta), k)
        i : np.ndarray
            Index results. Shape is (len(meta), k)
        """

        if os.path.exists(tree_fpath):
            tree = MetaManager.load_pickle_tree(tree_fpath)

        elif os.path.exists(meta_fpath):
            meta_source = pd.read_csv(meta_fpath)
            # pylint: disable=not-callable
            tree = cKDTree(meta_source[['latitude', 'longitude']])
            with open(tree_fpath, 'wb') as pkl:
                pickle.dump(tree, pkl)

        else:
            e = (
                'Missing both meta source tree file and meta source csv '
                'file: {}, {}'.format(tree_fpath, meta_fpath)
            )
            logger.error(e)
            raise FileNotFoundError(e)

        d, i = tree.query(meta[['latitude', 'longitude']].values, k=k)
        if len(i.shape) == 1:
            d = d.reshape((len(i), 1))
            i = i.reshape((len(i), 1))
        return d, i

    def _get_agg_method(self, var):
        """Get the aggregation method for a given variable.

        Parameters
        ----------
        var : str
            Variable name

        Returns
        -------
        method : function
            Aggregation method for the input var from the Aggregation class
            above.
        """
        return self.AGG_METHODS.get(var, self.DEFAULT_METHOD)

    def _get_args(self, var, i):
        """Get an argument list for a given variable and site.

        Parameters
        ----------
        var : str
            Variable name.
        i : int
            Site index number in current meta chunk.

        Returns
        -------
        args : list
            List of arguments
        """

        if 'dhi' in var:
            args = [var, i, self.fout]

        else:
            gid = self.meta_chunk.index.values[i]
            source = self.meta_chunk.iloc[i, :]['source']
            nn = self.data[source]['nn'][gid, :]
            w = self.data[source]['window']

            if 'fpath' in self.data[source]:
                data_fpath = self.data[source]['fpath']
            else:
                data_sub_dir = self.data[source]['data_sub_dir']
                data_fpath = self._get_fpath(
                    var, self.data_dir, data_sub_dir, source
                )

            if not os.path.exists(data_fpath) and '*' not in data_fpath:
                e = 'Could not find source data filepath for "{}": {}'.format(
                    var, data_fpath
                )
                logger.error(e)
                raise FileNotFoundError(e)

            args = [var, data_fpath, nn, w, self.time_index]

        if 'cld_' in var:
            args.append(gid)
            args.append(self.fout)

        return args

    def _agg_var_serial(self, var, method):
        """Aggregate one var for all sites in this chunk in parallel.

        Parameters
        ----------
        var : str
            Variable name being aggregated.
        method : function
            Aggregation method.

        Returns
        -------
        arr : np.ndarray
            Aggregated data with shape (t, n) where t is the final time index
            length and n is the number of sites in the current meta chunk.
        """

        arr = self._init_arr(var)

        for i in range(len(self.meta_chunk)):
            logger.debug('Kicking off site index #{}'.format(i))
            args = self._get_args(var, i)
            arr[:, i] = method(*args)

        return arr

    def _agg_var_parallel(self, var, method, max_workers=None):
        """Aggregate one var for all sites in this chunk in parallel.

        Parameters
        ----------
        var : str
            Variable name being aggregated.
        method : function
            Aggregation method.

        Returns
        -------
        arr : np.ndarray
            Aggregated data with shape (t, n) where t is the final time index
            length and n is the number of sites in the current meta chunk.
        """

        futures = {}
        arr = self._init_arr(var)

        loggers = ['farms', 'nsrdb']
        with SpawnProcessPool(loggers=loggers, max_workers=max_workers) as exe:
            logger.debug('Submitting futures...')
            for i in range(len(self.meta_chunk)):
                args = self._get_args(var, i)
                f = exe.submit(method, *args)
                futures[f] = i
            logger.debug('Finished submitting futures')

            for j, f in enumerate(as_completed(futures)):
                if (j + 1) % 1000 == 0:
                    logger.info(
                        'Futures completed: {} out of {}.'.format(
                            j + 1, len(futures)
                        )
                    )
                i = futures[f]
                arr[:, i] = f.result()

        return arr

    def write_output(self, arr, var):
        """Write aggregated output data to the final output file.

        Parameters
        ----------
        arr : np.ndarray
            Aggregated data with shape (t, n) where t is the final time index
            length and n is the number of sites in the current meta chunk.
        var : str
            Variable (dataset) name to write to.
        """
        logger.debug('Writing data for "{}" to: {}.'.format(var, self.fout))
        with Outputs(self.fout, mode='a') as out:
            out[var] = arr

    @classmethod
    def run_chunk(
        cls,
        data,
        data_dir,
        meta_dir,
        i_chunk,
        n_chunks,
        year=2018,
        ignore_dsets=None,
        max_workers=None,
        log_file='run_agg_chunk.log',
        log_level='DEBUG',
    ):
        """
        Parameters
        ----------
        data : dict
            Nested dictionary containing data on all NSRDB data sources
            (east, west, conus) and the final aggregated output.
        data_dir : str
            Root directory containing sub dirs with all data sources.
        meta_dir : str
            Directory containing meta and ckdtree files for each data source
            and the final aggregated output.
        i_chunk : int
            Single chunk index to process.
        n_chunks : int
            Number of chunks to process the meta data in.
        year : int
            Year being analyzed.
        ignore_dsets : list | None
            Source datasets to ignore (not aggregate). Optional.
        max_workers : int | None
            Number of workers to user. Runs serially if max_workers == 1
        log_file : str
            File to use for logging
        log_level : str | bool
            Flag to initialize a log file at a given log level.
            False will not init a logger.
        """

        init_logger(__name__, log_level=log_level, log_file=log_file)

        logger.info(
            'Working on site chunk {} out of {}'.format(i_chunk + 1, n_chunks)
        )

        m = cls(
            data,
            data_dir,
            meta_dir,
            year=year,
            i_chunk=i_chunk,
            n_chunks=n_chunks,
        )

        if ignore_dsets is None:
            ignore_dsets = []

        datasets = [
            d
            for d in m.dsets
            if 'dhi' not in d and 'cld_' not in d and d not in ignore_dsets
        ]
        delayed_datasets = [
            d
            for d in m.dsets
            if 'dhi' in d or ('cld_' in d and d not in ignore_dsets)
        ]
        n_var = len(datasets) + len(delayed_datasets)
        i_var = 0
        for dsets in [datasets, delayed_datasets]:
            for var in dsets:
                method = m._get_agg_method(var)
                i_var += 1
                logger.info(
                    'Aggregating variable "{}" '
                    '({} out of {}). Using method: {}'.format(
                        var, i_var, n_var, method
                    )
                )
                if max_workers == 1:
                    arr = m._agg_var_serial(var, method)
                else:
                    arr = m._agg_var_parallel(
                        var, method, max_workers=max_workers
                    )

                m.write_output(arr, var)

        logger.info('NSRDB aggregation complete!')

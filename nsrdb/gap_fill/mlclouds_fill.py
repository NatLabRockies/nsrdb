"""Cloud Properties filling using phgynn"""

import logging
import os
import shutil
import time
from concurrent.futures import as_completed
from typing import ClassVar
from warnings import warn

import numpy as np
import pandas as pd
import psutil
from farms import ICE_TYPES, WATER_TYPES
from mlclouds import LEG_MODEL_FPATH
from mlclouds.model.multi_step import MultiCloudsModel
from rex import MultiFileNSRDB
from rex.utilities.execution import SpawnProcessPool

from nsrdb.data_model.variable_factory import VarFactory
from nsrdb.file_handlers.outputs import Outputs
from nsrdb.file_handlers.resource import Resource
from nsrdb.gap_fill.cloud_fill import CloudGapFill

logger = logging.getLogger(__name__)


class MLCloudsFill:
    """
    Use the MLClouds algorithm with phygnn model to fill missing cloud data
    """

    DEFAULT_MODEL: ClassVar[dict] = {
        'cloud_type_model_path': None,
        'cloud_prop_model_path': LEG_MODEL_FPATH,
    }

    def __init__(
        self, h5_source, fill_all=False, model_path=None, var_meta=None
    ):
        """
        Parameters
        ----------
        h5_source : str
            Path to directory containing multi-file resource file sets.
            Available formats:
                /h5_dir/
                /h5_dir/prefix*suffix
        fill_all : bool
            Flag to fill all cloud properties for all timesteps where
            cloud_type is cloudy.
        model_path : dict | None
            kwargs for ``MultiCloudsModel.load`` method. Specifies
            ``cloud_prop_model_path`` for cloud property model and optionally
            ``cloud_type_model_path`` for a cloud type model. Each value is
            typically a fpath to a .pkl file with an accompanying .json file in
            the same directory. None will try to use the default model path
            from the mlclouds project directory.
        var_meta : str | pd.DataFrame | None
            CSV file or dataframe containing meta data for all NSRDB variables.
            Defaults to the NSRDB var meta csv in git repo.
        """

        self._dset_map = {}
        self._res_shape = None
        self._h5_source = h5_source
        self._fill_all = fill_all
        self._var_meta = var_meta
        if model_path is None:
            model_path = self.DEFAULT_MODEL

        logger.info(
            'Initializing MLCloudsFill with h5_source: {}'.format(
                self._h5_source
            )
        )
        logger.info(
            'Initializing MLCloudsFill with model: {}'.format(model_path)
        )
        logger.info(
            'MLCloudsFill filling all cloud properties: {}'.format(
                self._fill_all
            )
        )

        self._phygnn_model = MultiCloudsModel.load(**model_path)

        if self.h5_source is not None:
            with MultiFileNSRDB(self.h5_source) as res:
                self._dset_map = res.h5._dset_map
                self._res_shape = res.shape

    def preflight(self):
        """Run preflight checks and raise error if datasets are missing"""

        missing = []
        for dset in self._phygnn_model.feature_names:
            ignore = ('clear', 'ice_cloud', 'water_cloud', 'bad_cloud')
            if dset not in ignore and dset not in self._dset_map:
                missing.append(dset)

        if any(missing):
            msg = (
                'The following datasets were missing in the h5_source '
                'directory: {}'.format(missing)
            )
            logger.error(msg)
            raise FileNotFoundError(msg)

    @property
    def phygnn_model(self):
        """
        Pre-trained PhygnnModel instance

        Returns
        -------
        PhygnnModel
        """
        return self._phygnn_model

    @property
    def h5_source(self):
        """
        Path to directory containing multi-file resource file sets.
        Available formats:
            /h5_dir/
            /h5_dir/prefix*suffix

        Returns
        -------
        str
        """
        return self._h5_source

    @property
    def dset_map(self):
        """
        Mapping of datasets to .h5 files

        Returns
        -------
        dict
        """
        return self._dset_map

    def parse_feature_data(self, feature_data=None, col_slice=slice(None)):
        """
        Parse raw feature data from .h5 files (will have gaps!)

        Parameters
        ----------
        feature_data : dict | None
            Pre-loaded feature data to add to (optional). Keys are the feature
            names (nsrdb dataset names), values are 2D numpy arrays
            (time x sites). Any dsets already in this input won't be re-read.
        col_slice : slice
            Column slice of the resource data to work on. This is a result of
            chunking the columns to reduce memory load. Use slice(None) for
            no chunking.

        Returns
        -------
        feature_data : dict
            Raw feature data with gaps. keys are the feature names (nsrdb
            dataset names), values are 2D numpy arrays (time x sites).
        """

        logger.info('Loading feature data.')
        if feature_data is None:
            feature_data = {}

        dsets = (
            self._phygnn_model.feature_names + self._phygnn_model.label_names
        )

        with MultiFileNSRDB(self.h5_source) as res:
            for dset in dsets:
                if dset not in feature_data and dset in res.dsets:
                    logger.debug('Loading {} data'.format(dset))
                    feature_data[dset] = res[dset, :, col_slice]

        mem = psutil.virtual_memory()
        logger.info(
            'Feature data loaded for column slice {}. '
            'Memory utilization is {:.3f} GB out of {:.3f} GB total '
            '({:.2f}% used).'.format(
                col_slice,
                mem.used / 1e9,
                mem.total / 1e9,
                100 * mem.used / mem.total,
            )
        )

        return feature_data

    def init_clean_arrays(self):
        """Initialize a dict of numpy arrays for clean data."""
        arr = np.zeros(self._res_shape, dtype=np.float32)
        dsets = (
            'cloud_type',
            'cld_press_acha',
            'cld_opd_dcomp',
            'cld_reff_dcomp',
        )
        clean_arrays = {d: arr.copy() for d in dsets}
        fill_flag_array = arr.copy().astype(np.uint8)
        return clean_arrays, fill_flag_array

    @staticmethod
    def clean_array(dset, array):
        """Clean a dataset array using temporal nearest neighbor interpolation.

        Parameters
        ----------
        dset : str
            NSRDB dataset name
        array : np.ndarray
            2D (time x sites) float numpy array of data for dset. Missing
            values should be set to NaN.

        Returns
        -------
        array : np.ndarray
            2D (time x sites) float numpy array of data for dset. Missing
            values should be filled
        """

        # attempt to clean sites with mostly NaN values. This has to happen
        # before the df.interpolate() call.
        bad = np.isnan(array)
        any_bad = bad.any()
        bad_cols = (~bad).sum(axis=0) < 3
        if bad.all():
            msg = (
                f'Feature dataset "{dset}" has all NaN data! Filling with '
                'zeros.'
            )
            logger.warning(msg)
            warn(msg)
            array[:] = 0
        elif bad_cols.any():
            mean_impute = np.nanmean(array, axis=0)
            count = bad_cols.sum()
            msg = (
                'Feature dataset "{}" has {} columns with nearly all NaN '
                'values out of {} ({:.2f}%) ({} NaN values out of '
                '{} total {:.2f}%). Filling with mean values by site.'.format(
                    dset,
                    count,
                    array.shape[1],
                    100 * count / array.shape[1],
                    bad.sum(),
                    array.size,
                    100 * bad.sum() / array.size,
                )
            )
            logger.warning(msg)
            warn(msg)
            array[:, bad_cols] = mean_impute[bad_cols]

        # attempt to fill all remaining NaN values that should be scattered
        # throughout (not full sites missing data)
        if any_bad:
            array = (
                pd.DataFrame(array)
                .interpolate('nearest')
                .ffill()
                .bfill()
                .values
            )

        # Fill any persistent NaN values with the global mean (these are
        # usually sites that have no valid data at all)
        bad = np.isnan(array)
        bad_cols = (~bad).sum(axis=0) < 3
        if bad.any() or bad_cols.any():
            mean_impute = np.nanmean(array)
            msg = (
                'There were {} observations (out of {}) that have '
                'persistent NaN values and {} sites (out of {}) that '
                'still have all NaN values that could not be '
                'cleaned for {}. Filling with global mean value of {}'.format(
                    bad.sum(),
                    bad.size,
                    bad_cols.sum(),
                    array.shape[1],
                    dset,
                    mean_impute,
                )
            )
            logger.warning(msg)
            warn(msg)
            array[bad] = mean_impute

        return array

    def clean_feature_data(
        self, feature_raw, fill_flag, sza_lim=90, max_workers=1
    ):
        """
        Clean feature data

        Parameters
        ----------
        feature_raw : dict
            Raw feature data with gaps. keys are the feature names (nsrdb
            dataset names), values are 2D numpy arrays (time x sites).
        fill_flag : np.ndarray
            Integer array of flags showing what data was filled and why.
        sza_lim : int, optional
            Solar zenith angle limit below which missing cloud property data
            will be gap filled. By default 90 to fill all missing daylight data
        max_workers : None | int
            Maximum workers to clean data in parallel. 1 is serial and None
            uses all available workers.

        Returns
        -------
        feature_data : dict
            Clean feature data without gaps. keys are the feature names
            (nsrdb dataset names), values are 2D numpy arrays (time x sites).
        fill_flag : np.ndarray
            Integer array of flags showing what data was filled and why.
        """

        t0 = time.time()
        feature_data = feature_raw.copy()

        day = feature_data['solar_zenith_angle'] < sza_lim

        cloud_type = feature_data['cloud_type']
        assert cloud_type.shape == fill_flag.shape
        # fill flag 1 and 2 are the missing cloud type flags
        day_missing_ctype = day & (
            np.isin(fill_flag, (1, 2)) | (cloud_type < 0)
        )

        mask = cloud_type < 0
        full_missing_ctype_mask = mask.all(axis=0)
        if any(full_missing_ctype_mask) or mask.any():
            msg = (
                'There are {} missing cloud type observations '
                'out of {} including {} with full missing ctype. This '
                'needs to be resolved using the fill_ctype_press() '
                'method before this step.'.format(
                    mask.sum(), mask.size, full_missing_ctype_mask.sum()
                )
            )
            logger.error(msg)
            raise RuntimeError(msg)

        feature_data['cld_opd_dcomp'] = np.nan_to_num(
            feature_data['cld_opd_dcomp'], nan=-1.0
        )
        feature_data['cld_reff_dcomp'] = np.nan_to_num(
            feature_data['cld_reff_dcomp'], nan=-1.0
        )

        cloudy = np.isin(cloud_type, ICE_TYPES + WATER_TYPES)
        day_clouds = day & cloudy
        day_missing_opd = day_clouds & (feature_data['cld_opd_dcomp'] <= 0)
        day_missing_reff = day_clouds & (feature_data['cld_reff_dcomp'] <= 0)

        mask_fill_flag_opd = day_missing_opd & (fill_flag == 0)
        mask_fill_flag_reff = day_missing_reff & (fill_flag == 0)
        mask_all_bad_opd = (day_missing_opd | ~day).all(axis=0) & (
            fill_flag < 2
        ).all(axis=0)
        mask_all_bad_reff = (day_missing_reff | ~day).all(axis=0) & (
            fill_flag < 2
        ).all(axis=0)
        fill_flag[mask_fill_flag_opd] = 7  # fill_flag=7 is mlcloud fill
        fill_flag[mask_fill_flag_reff] = 7  # fill_flag=7 is mlcloud fill
        fill_flag[:, mask_all_bad_opd] = 4
        fill_flag[:, mask_all_bad_reff] = 4

        logger.info(
            '{:.2f}% of timesteps are daylight'.format(
                100 * day.sum() / (day.shape[0] * day.shape[1])
            )
        )
        logger.info(
            '{:.2f}% of daylight timesteps are cloudy'.format(
                100 * day_clouds.sum() / day.sum()
            )
        )
        logger.info(
            '{:.2f}% of daylight timesteps are missing cloud type'.format(
                100 * day_missing_ctype.sum() / day.sum()
            )
        )
        logger.info(
            '{:.2f}% of cloudy daylight timesteps are missing cloud '
            'opd'.format(100 * day_missing_opd.sum() / day_clouds.sum())
        )
        logger.info(
            '{:.2f}% of cloudy daylight timesteps are missing cloud '
            'reff'.format(100 * day_missing_reff.sum() / day_clouds.sum())
        )

        mask = feature_data['cld_opd_dcomp'] <= 0
        feature_data['cld_opd_dcomp'][mask] = np.nan
        feature_data['cld_opd_dcomp'][~cloudy] = 0.0

        mask = feature_data['cld_reff_dcomp'] <= 0
        feature_data['cld_reff_dcomp'][mask] = np.nan
        feature_data['cld_reff_dcomp'][~cloudy] = 0.0

        logger.debug('Column NaN values:')
        for c, d in feature_data.items():
            pnan = 100 * np.isnan(d).sum() / (d.shape[0] * d.shape[1])
            logger.debug('\t"{}" has {:.2f}% NaN values'.format(c, pnan))

        logger.debug('Interpolating feature data using nearest neighbor.')

        if max_workers == 1:
            for c, d in feature_data.items():
                if c not in self.phygnn_model.label_names:
                    feature_data[c] = self.clean_array(c, d)
        else:
            logger.info(
                'Interpolating feature data with {} max workers'.format(
                    max_workers
                )
            )
            futures = {}
            loggers = ['nsrdb', 'rex', 'phygnn']
            with SpawnProcessPool(
                loggers=loggers, max_workers=max_workers
            ) as exe:
                for c, d in feature_data.items():
                    if c not in self.phygnn_model.label_names:
                        future = exe.submit(self.clean_array, c, d)
                        futures[future] = c

                for future in as_completed(futures):
                    c = futures[future]
                    feature_data[c] = future.result()

        logger.debug('Feature data interpolation is complete.')

        assert ~(feature_data['cloud_type'] < 0).any()
        assert ~any(np.isnan(d).any() for d in feature_data.values())
        assert ~(cloudy & (feature_data['cld_opd_dcomp'] <= 0)).any()
        assert ~(cloudy & (feature_data['cld_reff_dcomp'] <= 0)).any()

        logger.debug('Adding feature flag')
        cloud_type = feature_data['cloud_type']
        day_ice_clouds = day & np.isin(cloud_type, ICE_TYPES)
        day_water_clouds = day & np.isin(cloud_type, WATER_TYPES)
        day_clouds_bad_ctype = (
            day_ice_clouds | day_water_clouds
        ) & day_missing_ctype

        flag = np.full(day_ice_clouds.shape, 'night', dtype=object)
        flag[day] = 'clear'
        flag[day_ice_clouds] = 'ice_cloud'
        flag[day_water_clouds] = 'water_cloud'
        flag[day_clouds_bad_ctype] = 'bad_cloud'
        flag[day_missing_opd] = 'bad_cloud'
        flag[day_missing_reff] = 'bad_cloud'
        feature_data['flag'] = flag
        logger.debug(
            'Created the "flag" dataset with the following unique '
            'values: {}'.format(np.unique(flag))
        )

        mem = psutil.virtual_memory()
        logger.debug(
            'Cleaned feature data dict has these keys: {}'.format(
                feature_data.keys()
            )
        )
        logger.debug(
            'Cleaned feature data dict values have these shapes: {}'.format([
                d.shape for d in feature_data.values()
            ])
        )
        logger.debug(
            'Feature flag column has these values: {}'.format(
                np.unique(feature_data['flag'])
            )
        )
        logger.info('Cleaning took {:.1f} seconds'.format(time.time() - t0))
        logger.info(
            'Memory utilization is '
            '{:.3f} GB out of {:.3f} GB total ({:.2f}% used).'.format(
                mem.used / 1e9, mem.total / 1e9, 100 * mem.used / mem.total
            )
        )

        return feature_data, fill_flag

    def archive_cld_properties(self):
        """
        Archive original cloud property (cld_*) .h5 files. This method creates
        .tmp files in a ./raw/ sub directory. mark_complete_archived_files()
        should be run at the end to remove the .tmp designation. This will
        signify that the cloud fill was completed successfully.
        """
        cld_dsets = [
            'cld_opd_dcomp',
            'cld_reff_dcomp',
            'cld_press_acha',
            'cloud_type',
        ]
        for dset in [ds for ds in cld_dsets if ds in self.dset_map]:
            src_fpath = self.dset_map[dset]
            src_dir, f_name = os.path.split(src_fpath)
            dst_dir = os.path.join(src_dir, 'raw')
            if not os.path.exists(dst_dir):
                os.makedirs(dst_dir)

            dst_fpath = os.path.join(dst_dir, f_name)
            dst_fpath_tmp = dst_fpath + '.tmp'
            logger.debug('Archiving {} to {}'.format(src_fpath, dst_fpath_tmp))
            if os.path.exists(dst_fpath):
                msg = (
                    'A raw cloud file already exists, this suggests '
                    'MLClouds gap fill has already been run: {}'.format(
                        dst_fpath
                    )
                )
                logger.error(msg)
                raise RuntimeError(msg)
            if os.path.exists(dst_fpath_tmp):
                # don't overwrite the tmp file, the original may have been
                # manipulated by a failed mlclouds job.
                logger.debug(
                    'Archive file exists, not overwriting: {}'.format(
                        dst_fpath_tmp
                    )
                )
            else:
                shutil.copy(src_fpath, dst_fpath_tmp)

    def mark_complete_archived_files(self):
        """Remove the .tmp marker from the archived files once MLCloudsFill
        is complete"""
        cld_dsets = [
            'cld_opd_dcomp',
            'cld_reff_dcomp',
            'cld_press_acha',
            'cloud_type',
        ]
        for dset in cld_dsets:
            fpath = self.dset_map[dset]
            src_dir, f_name = os.path.split(fpath)
            raw_path = os.path.join(src_dir, 'raw', f_name)
            logger.debug('Renaming .tmp raw file to {}'.format(raw_path))
            if os.path.exists(raw_path + '.tmp'):
                os.rename(raw_path + '.tmp', raw_path)
            else:
                msg = (
                    'Something went wrong. The .tmp file created at the '
                    'beginning of MLCloudsFill no longer exists: {}'.format(
                        raw_path + '.tmp'
                    )
                )
                logger.error(msg)
                raise FileNotFoundError(msg)

    def predict_cld_properties(
        self, feature_data, col_slice=None, low_mem=False
    ):
        """
        Predict cloud properties with phygnn

        Parameters
        ----------
        feature_data : dict
            Clean feature data without gaps. keys are the feature names
            (nsrdb dataset names), values are 2D numpy arrays (time x sites).
        col_slice : slice
            Column slice of the resource data to work on. This is a result of
            chunking the columns to reduce memory load. Just used for logging
            in this method.
        low_mem : bool
            Option to run predictions in low memory mode. Typically the
            memory bloat during prediction is:
            (n_time x n_sites x n_nodes_per_layer). low_mem=True will
            reduce this to (1000 x n_nodes_per_layer)

        Returns
        -------
        predicted_data : dict
            Dictionary of predicted cloud properties. Keys are nsrdb dataset
            names, values are 2D arrays of phygnn-predicted values
            (time x sites).
        """

        L = feature_data['flag'].shape[0] * feature_data['flag'].shape[1]
        cols = [
            k
            for k in feature_data
            if k in self.phygnn_model.feature_names or k == 'flag'
        ]
        feature_df = pd.DataFrame(index=np.arange(L), columns=cols)
        for dset, arr in feature_data.items():
            if dset in feature_df:
                feature_df[dset] = arr.flatten(order='F')

        # Predict on night timesteps as if they were clear
        # the phygnn model wont be trained with the night category
        # so this is an easy way to keep the full data shape and
        # cooperate with the feature names that phygnn expects
        night_mask = feature_df['flag'] == 'night'
        feature_df.loc[night_mask, 'flag'] = 'clear'

        logger.debug(
            'Predicting gap filled cloud data for column slice {}'.format(
                col_slice
            )
        )
        predict_feats = set(feature_df.columns).intersection(
            self.phygnn_model.input_feature_names
        )
        predict_feats = feature_df[list(predict_feats)]
        if not low_mem:
            labels = self.phygnn_model.predict(predict_feats, table=False)
        else:
            len_df = len(predict_feats)
            chunks = np.array_split(
                np.arange(len_df), int(np.ceil(len_df / 1000))
            )
            labels = []
            for index_chunk in chunks:
                sub = predict_feats.iloc[index_chunk]
                labels.append(self.phygnn_model.predict(sub, table=False))
            labels = np.concatenate(labels, axis=0)

        mem = psutil.virtual_memory()
        logger.debug(
            'Prediction complete for column slice {}. Memory '
            'utilization is {:.3f} GB out of {:.3f} GB total '
            '({:.2f}% used).'.format(
                col_slice,
                mem.used / 1e9,
                mem.total / 1e9,
                100 * mem.used / mem.total,
            )
        )
        logger.debug('Label data shape: {}'.format(labels.shape))

        shape = feature_data['flag'].shape
        predicted_data = {}
        for i, dset in enumerate(self.phygnn_model.output_names):
            logger.debug('Reshaping predicted {} to {}'.format(dset, shape))
            predicted_data[dset] = labels[:, i].reshape(shape, order='F')

        for dset, arr in predicted_data.items():
            nnan = np.isnan(arr).sum()
            ntot = arr.shape[0] * arr.shape[1]
            logger.debug(
                'Raw predicted data for {} for column slice {} '
                'has mean: {:.2f}, median: {:.2f}, range: '
                '({:.2f}, {:.2f}) and {} NaN values out of '
                '{} ({:.2f}%)'.format(
                    dset,
                    col_slice,
                    np.nanmean(arr),
                    np.median(arr),
                    np.nanmin(arr),
                    np.nanmax(arr),
                    nnan,
                    ntot,
                    100 * nnan / ntot,
                )
            )

        return predicted_data

    def fill_bad_cld_properties(
        self, predicted_data, feature_data, col_slice=None
    ):
        """
        Fill bad cloud properties in the feature data from predicted data

        Parameters
        ----------
        predicted_data : dict
            Dictionary of predicted cloud properties. Keys are nsrdb dataset
            names, values are 2D arrays of phygnn-predicted values
            (time x sites).
        feature_data : dict
            Clean feature data without gaps. keys are the feature names
            (nsrdb dataset names), values are 2D numpy arrays (time x sites).
        col_slice : slice
            Column slice of the resource data to work on. This is a result of
            chunking the columns to reduce memory load. Just used for logging
            in this method.

        Returns
        -------
        filled_data : dict
            Dictionary of filled cloud properties. Keys are nsrdb dataset
            names, values are 2D arrays of phygnn-predicted values
            (time x sites). The filled data is a combination of the input
            predicted_data and feature_data. The datasets in the predicted_data
            input are used to fill the feature_data input where:
            (feature_data['flag'] == "bad_cloud")
        """

        fill_mask = feature_data['flag'] == 'bad_cloud'
        night_mask = feature_data['flag'] == 'night'
        clear_mask = feature_data['flag'] == 'clear'
        cloud_mask = (feature_data['flag'] != 'night') & (
            feature_data['flag'] != 'clear'
        )
        logger.debug(
            'Final fill mask has {} bad clouds, {} night timesteps, '
            '{} clear timesteps, {} timesteps that are either night '
            'or clear, and {} cloudy timesteps out of {} '
            'total observations.'.format(
                fill_mask.sum(),
                night_mask.sum(),
                clear_mask.sum(),
                (clear_mask | night_mask).sum(),
                cloud_mask.sum(),
                fill_mask.size,
            )
        )

        if fill_mask.sum() == 0:
            msg = (
                'No "bad_cloud" flags were detected in the feature_data '
                '"flag" dataset. Something went wrong! '
                'The cloud data is never perfect...'
            )
            logger.warning(msg)
            warn(msg)

        if self._fill_all:
            logger.debug(
                'Filling {} values (all cloudy timesteps) using '
                'MLClouds predictions for column slice {}'.format(
                    np.sum(cloud_mask), col_slice
                )
            )
        else:
            logger.debug(
                'Filling {} values using MLClouds predictions for '
                'column slice {}'.format(np.sum(fill_mask), col_slice)
            )

        filled_data = {}
        for dset, arr in predicted_data.items():
            varobj = VarFactory.get_base_handler(dset, var_meta=self._var_meta)
            arr = np.maximum(arr, 0.01)
            arr = np.minimum(arr, varobj.physical_max)

            cld_data = feature_data[dset]
            if self._fill_all:
                cld_data[cloud_mask] = arr[cloud_mask]
            else:
                cld_data[fill_mask] = arr[fill_mask]

            cld_data[night_mask | clear_mask] = 0
            filled_data[dset] = cld_data

            nnan = np.isnan(filled_data[dset]).sum()
            ntot = filled_data[dset].shape[0] * filled_data[dset].shape[1]
            logger.debug(
                'Final cleaned data for {} for column slice {} '
                'has mean: {:.2f}, median: {:.2f}, range: '
                '({:.2f}, {:.2f}) and {} NaN values out of '
                '{} ({:.2f}%)'.format(
                    dset,
                    col_slice,
                    np.nanmean(filled_data[dset]),
                    np.median(filled_data[dset]),
                    np.nanmin(filled_data[dset]),
                    np.nanmax(filled_data[dset]),
                    nnan,
                    ntot,
                    100 * nnan / ntot,
                )
            )

            if nnan > 0:
                msg = (
                    'Gap filled cloud property "{}" still had '
                    '{} NaN values!'.format(dset, nnan)
                )
                logger.error(msg)
                raise RuntimeError(msg)

        return filled_data

    @staticmethod
    def fill_ctype_press(h5_source, col_slice=slice(None)):
        """Fill cloud type and pressure using simple temporal nearest neighbor.

        Parameters
        ----------
        h5_source : str
            Path to directory containing multi-file resource file sets.
            Available formats:
                /h5_dir/
                /h5_dir/prefix*suffix
        col_slice : slice
            Column slice of the resource data to work on. This is a result of
            chunking the columns to reduce memory load. Use slice(None) for
            no chunking.

        Returns
        -------
        cloud_type : np.ndarray
            2D array (time x sites) of gap-filled cloud type data.
        cloud_pres : np.ndarray
            2D array (time x sites) of gap-filled cld_press_acha data.
        sza : np.ndarray
            2D array (time x sites) of solar zenith angle data.
        fill_flag : np.ndarray
            Integer array of flags showing what data was filled and why.
        """

        fill_flag = None
        with MultiFileNSRDB(h5_source) as f:
            cloud_type = f['cloud_type', :, col_slice]
            cloud_pres = f['cld_press_acha', :, col_slice]
            sza = f['solar_zenith_angle', :, col_slice]

        cloud_type, fill_flag = CloudGapFill.fill_cloud_type(
            cloud_type, fill_flag=fill_flag
        )

        cloud_pres, fill_flag = CloudGapFill.fill_cloud_prop(
            'cld_press_acha',
            cloud_pres,
            cloud_type,
            sza,
            fill_flag=fill_flag,
            cloud_type_is_clean=True,
        )

        return cloud_type, cloud_pres, sza, fill_flag

    def write_filled_data(self, filled_data, col_slice=slice(None)):
        """Write gap filled cloud data to disk

        Parameters
        ----------
        filled_data : dict
            Dictionary of filled cloud properties. Keys are nsrdb dataset
            names, values are 2D arrays of phygnn-predicted values
            (time x sites). The filled data is a combination of the input
            predicted_data and feature_data. The datasets in the predicted_data
            input are used to fill the feature_data input where:
            (feature_data['flag'] == "bad_cloud")
        col_slice : slice
            Column slice of the resource data to work on. This is a result of
            chunking the columns to reduce memory load. Use slice(None) for
            no chunking.
        """

        for dset, arr in filled_data.items():
            fpath = self.dset_map[dset]
            with Outputs(fpath, mode='a') as f:
                logger.info(
                    'Writing filled "{}" to: {}'.format(
                        dset, os.path.basename(fpath)
                    )
                )
                f[dset, :, col_slice] = arr
                logger.debug('Finished writing "{}".'.format(dset))

    def write_fill_flag(self, fill_flag, col_slice=slice(None)):
        """Write the fill flag dataset to its daily file next to the cloud
        property files.

        Parameters
        ----------
        fill_flag : np.ndarray
            Integer array of flags showing what data was filled and why.
        col_slice : slice
            Column slice of the resource data to work on. This is a result of
            chunking the columns to reduce memory load. Use slice(None) for
            no chunking.
        """
        fpath_opd = self._dset_map['cld_opd_dcomp']
        fpath = fpath_opd.replace('cld_opd_dcomp', 'cloud_fill_flag')
        var_obj = VarFactory.get_base_handler(
            'cloud_fill_flag', var_meta=self._var_meta
        )

        with Resource(fpath_opd) as res:
            ti = res.time_index
            meta = res.meta

        logger.info(
            'Writing cloud_fill_flag to: {}'.format(os.path.basename(fpath))
        )

        init_dset = False
        if os.path.exists(fpath):
            with Outputs(fpath, mode='r') as fout:
                if 'cloud_fill_flag' not in fout:
                    init_dset = True
        else:
            init_dset = True

        try:
            if init_dset:
                with Outputs(fpath, mode='w') as fout:
                    fout.time_index = ti
                    fout.meta = meta
                    init_data = np.zeros(
                        (len(ti), len(meta)), dtype=var_obj.final_dtype
                    )
                    fout._add_dset(
                        dset_name='cloud_fill_flag',
                        dtype=var_obj.final_dtype,
                        data=init_data,
                        chunks=var_obj.chunks,
                        attrs=var_obj.attrs,
                    )

            with Outputs(fpath, mode='a') as fout:
                fout['cloud_fill_flag', :, col_slice] = fill_flag
        except Exception as e:
            msg = 'Could not write col_slice {} to file: "{}"'.format(
                col_slice, fpath
            )
            logger.exception(msg)
            raise OSError from e

        logger.debug('Write complete')
        logger.info('Final cloud_fill_flag counts:')
        ntot = fill_flag.shape[0] * fill_flag.shape[1]
        for n in range(10):
            count = (fill_flag == n).sum()
            logger.info(
                '\tFlag {} has {} counts out of {} ({:.2f}%)'.format(
                    n, count, ntot, 100 * count / ntot
                )
            )

    @classmethod
    def prep_chunk(
        cls,
        h5_source,
        model_path=None,
        var_meta=None,
        sza_lim=90,
        col_slice=slice(None),
    ):
        """Prepare a column chunk (slice) of data for phygnn prediction.

        Parameters
        ----------
        h5_source : str
            Path to directory containing multi-file resource file sets.
            Available formats:
                /h5_dir/
                /h5_dir/prefix*suffix
        model_path : str | None
            Directory to load phygnn model from. This is typically a fpath to
            a .pkl file with an accompanying .json file in the same directory.
            None will try to use the default model path from the mlclouds
            project directory.
        var_meta : str | pd.DataFrame | None
            CSV file or dataframe containing meta data for all NSRDB variables.
            Defaults to the NSRDB var meta csv in git repo.
        sza_lim : int, optional
            Solar zenith angle limit below which missing cloud property data
            will be gap filled. By default 90 to fill all missing daylight data
        col_slice : slice
            Column slice of the resource data to work on. This is a result of
            chunking the columns to reduce memory load. Use slice(None) for
            no chunking.

        Returns
        -------
        feature_data : dict
            Clean feature data without gaps. keys are the feature names
            (nsrdb dataset names), values are 2D numpy arrays (time x sites).
            This is just for the col_slice being worked on.
        clean_data : dict
            Dictionary of filled cloud properties. Keys are nsrdb dataset
            names, values are 2D arrays (time x sites) and have
            nearest-neighbor cleaned values for cloud pressure and type
            This is just for the col_slice being worked on.
        fill_flag : np.ndarray
            Integer array of flags showing what data was filled and why.
            This is just for the col_slice being worked on.
        """
        obj = cls(h5_source, model_path=model_path, var_meta=var_meta)

        logger.debug(
            'Preparing data for MLCloudsFill for column slice {}'.format(
                col_slice
            )
        )
        ctype, cpres, sza, fill_flag = obj.fill_ctype_press(
            obj.h5_source, col_slice=col_slice
        )
        clean_data = {'cloud_type': ctype, 'cld_press_acha': cpres}
        feature_data = {
            'cloud_type': ctype,
            'cld_press_acha': cpres,
            'solar_zenith_angle': sza,
        }
        feature_data = obj.parse_feature_data(
            feature_data=feature_data, col_slice=col_slice
        )
        feature_data, fill_flag = obj.clean_feature_data(
            feature_data, fill_flag, sza_lim=sza_lim
        )
        logger.debug(
            'Completed MLClouds data prep for column slice {}'.format(
                col_slice
            )
        )

        return feature_data, clean_data, fill_flag

    def process_chunk(
        self,
        i_features,
        i_clean,
        i_flag,
        col_slice,
        clean_data,
        fill_flag,
        low_mem=False,
    ):
        """Use cleaned and prepared data to run phygnn predictions and create
        final filled data for a single column chunk.

        Parameters
        ----------
        i_features : dict
            Clean feature data without gaps. keys are the feature names
            (nsrdb dataset names), values are 2D numpy arrays (time x sites).
            This is just for a single column chunk (col_slice).
        i_clean : dict
            Dictionary of filled cloud properties. Keys are nsrdb dataset
            names, values are 2D arrays (time x sites) of phygnn-predicted
            values (cloud opd and reff) or nearest-neighbor cleaned values
            (cloud pressure and type).
            This is just for a single column chunk (col_slice).
        i_flag : np.ndarray
            Integer array of flags showing what data was filled and why.
            This is just for a single column chunk (col_slice).
        col_slice : slice
            Column slice of the resource data to work on. This is a result of
            chunking the columns to reduce memory load.
        clean_data : dict
            Dictionary of filled cloud properties. Keys are nsrdb dataset
            names, values are 2D arrays (time x sites).
            This is for ALL chunks (full resource shape).
        fill_flag : np.ndarray
            Integer array of flags showing what data was filled and why.
            This is for ALL chunks (full resource shape).
        low_mem : bool
            Option to run predictions in low memory mode. Typically the
            memory bloat during prediction is:
            (n_time x n_sites x n_nodes_per_layer). low_mem=True will
            reduce this to (1000 x n_nodes_per_layer)

        Returns
        -------
        clean_data : dict
            Dictionary of filled cloud properties. Keys are nsrdb dataset
            names, values are 2D arrays (time x sites). This has been updated
            with phygnn-predicted values (cloud opd and reff) or
            nearest-neighbor cleaned values (cloud pressure and type)
            This is for ALL chunks (full resource shape).
        fill_flag : np.ndarray
            Integer array of flags showing what data was filled and why.
            This is for ALL chunks (full resource shape).
        """

        i_predicted = self.predict_cld_properties(
            i_features, col_slice=col_slice, low_mem=low_mem
        )
        i_filled = self.fill_bad_cld_properties(
            i_predicted, i_features, col_slice=col_slice
        )
        i_clean.update(i_filled)

        fill_flag[:, col_slice] = i_flag
        for k, v in clean_data.items():
            v[:, col_slice] = i_clean[k]

        return clean_data, fill_flag

    @classmethod
    def clean_data_model(
        cls,
        data_model,
        fill_all=False,
        model_path=None,
        var_meta=None,
        sza_lim=90,
        low_mem=False,
    ):
        """Run the MLCloudsFill process on data in-memory in an nsrdb
        data model object.

        Parameters
        ----------
        data_model : nsrdb.data_model.DataModel
            DataModel object with processed source data (cloud data + ancillary
            processed onto the nsrdb grid).
        fill_all : bool
            Flag to fill all cloud properties for all timesteps where
            cloud_type is cloudy.
        model_path : str | None
            Directory to load phygnn model from. This is typically a fpath to
            a .pkl file with an accompanying .json file in the same directory.
            None will try to use the default model path from the mlclouds
            project directory.
        var_meta : str | pd.DataFrame | None
            CSV file or dataframe containing meta data for all NSRDB variables.
            Defaults to the NSRDB var meta csv in git repo.
        sza_lim : int, optional
            Solar zenith angle limit below which missing cloud property data
            will be gap filled. By default 90 to fill all missing daylight data
        low_mem : bool
            Option to run predictions in low memory mode. Typically the
            memory bloat during prediction is:
            (n_time x n_sites x n_nodes_per_layer). low_mem=True will
            reduce this to (1000 x n_nodes_per_layer)

        Returns
        -------
        data_model : nsrdb.data_model.DataModel
            DataModel object with processed source data (cloud data + ancillary
            processed onto the nsrdb grid). The cloud property datasets
            (cloud_type, cld_opd_dcomp, cld_reff_dcomp, cloud_fill_flag) are
            now cleaned.
        """

        obj = cls(
            None, fill_all=fill_all, model_path=model_path, var_meta=var_meta
        )

        logger.info('Preparing data for MLCloudsFill to clean data model')

        ctype = data_model['cloud_type']
        sza = data_model['solar_zenith_angle']

        ctype, fill_flag = CloudGapFill.fill_cloud_type(ctype, fill_flag=None)
        feature_data = {'cloud_type': ctype, 'solar_zenith_angle': sza}

        logger.info('Loading feature data.')
        dsets = obj._phygnn_model.feature_names + obj._phygnn_model.label_names

        for dset in dsets:
            if dset not in feature_data and dset in data_model:
                feature_data[dset] = data_model[dset]

        mem = psutil.virtual_memory()
        logger.info(
            'Feature data loaded for data model cleaning. '
            'Memory utilization is {:.3f} GB out of {:.3f} GB total '
            '({:.2f}% used).'.format(
                mem.used / 1e9, mem.total / 1e9, 100 * mem.used / mem.total
            )
        )

        feature_data, fill_flag = obj.clean_feature_data(
            feature_data, fill_flag, sza_lim=sza_lim
        )
        logger.info('Completed MLClouds data prep')

        predicted = obj.predict_cld_properties(feature_data, low_mem=low_mem)
        filled = obj.fill_bad_cld_properties(predicted, feature_data)

        feature_data['cloud_fill_flag'] = fill_flag
        for k, v in feature_data.items():
            logger.info(
                'Sending cleaned feature dataset "{}" to data model '
                'with shape {}'.format(k, v.shape)
            )
            data_model[k] = v
        for k, v in filled.items():
            logger.info(
                'Sending cleaned cloud property dataset "{}" to '
                'data model with shape {}'.format(k, v.shape)
            )
            data_model[k] = v

        logger.info('Finished MLClouds fill of data model object.')

        return data_model

    @classmethod
    def merra_clouds(
        cls,
        h5_source,
        var_meta=None,
        merra_fill_flag=8,
        fill_all=False,
        model_path=None,
    ):
        """Quick check to see if cloud data is from a merra source in which
        case it should be gap-free and cloud_fill_flag will be written with all
        8's

        Parameters
        ----------
        h5_source : str
            Path to directory containing multi-file resource file sets.
            Available formats:
                /h5_dir/
                /h5_dir/prefix*suffix
        var_meta : str | pd.DataFrame | None
            CSV file or dataframe containing meta data for all NSRDB variables.
            Defaults to the NSRDB var meta csv in git repo.
        merra_fill_flag : int
            Integer fill flag representing where merra data was used as source
            cloud data.
        fill_all : bool
            Flag to fill all cloud properties for all timesteps where
            cloud_type is cloudy.
        model_path : str | None
            Directory to load phygnn model from. This is typically a fpath to
            a .pkl file with an accompanying .json file in the same directory.
            None will try to use the default model path from the mlclouds
            project directory.

        Returns
        -------
        is_merra : bool
            Flag that is True if cloud data is from merra
        """

        mlclouds = cls(
            h5_source,
            var_meta=var_meta,
            model_path=model_path,
            fill_all=fill_all,
        )

        with MultiFileNSRDB(h5_source) as res:
            attrs = res.attrs.get('cld_opd_dcomp', {})

        if 'merra' in attrs.get('data_source', '').lower():
            logger.info(
                'Found cloud data from MERRA2 for source: {}'.format(h5_source)
            )
            fill_flag_arr = mlclouds.init_clean_arrays()[1]
            fill_flag_arr[:] = merra_fill_flag
            mlclouds.write_fill_flag(fill_flag_arr)
            return True

        return False

    @classmethod
    def run(
        cls,
        h5_source,
        fill_all=False,
        model_path=None,
        var_meta=None,
        sza_lim=90,
        col_chunk=None,
        max_workers=None,
        low_mem=False,
    ):
        """
        Fill cloud properties using phygnn predictions. Original files will be
        archived to a new "raw/" sub-directory

        Parameters
        ----------
        h5_source : str
            Path to directory containing multi-file resource file sets.
            Available formats:
                /h5_dir/
                /h5_dir/prefix*suffix
        fill_all : bool
            Flag to fill all cloud properties for all timesteps where
            cloud_type is cloudy.
        model_path : str | None
            Directory to load phygnn model from. This is typically a fpath to
            a .pkl file with an accompanying .json file in the same directory.
            None will try to use the default model path from the mlclouds
            project directory.
        var_meta : str | pd.DataFrame | None
            CSV file or dataframe containing meta data for all NSRDB variables.
            Defaults to the NSRDB var meta csv in git repo.
        sza_lim : int, optional
            Solar zenith angle limit below which missing cloud property data
            will be gap filled. By default 90 to fill all missing daylight data
        col_chunk : None | int
            Optional chunking method to gap fill one column chunk at a time
            to reduce memory requirements. If provided, this should be an
            integer specifying how many columns to work on at one time.
        max_workers : None | int
            Maximum workers for running mlclouds in parallel. 1 is serial and
            None uses all available workers.
        low_mem : bool
            Option to run predictions in low memory mode. Typically the
            memory bloat during prediction is:
            (n_time x n_sites x n_nodes_per_layer). low_mem=True will
            reduce this to (1000 x n_nodes_per_layer)
        """

        logger.info(
            f'Running MLCloudsFill with h5_source: {h5_source}. Running '
            f'MLCloudsFill with model: {model_path}. Running MLCloudsFill '
            f'with col_chunk: {col_chunk}.'
        )
        obj = cls(
            h5_source,
            fill_all=fill_all,
            model_path=model_path,
            var_meta=var_meta,
        )
        obj.preflight()
        obj.archive_cld_properties()
        clean_data, fill_flag = obj.init_clean_arrays()

        if col_chunk is None:
            slices = [slice(None)]
            logger.info(
                'MLClouds gap fill is being run without col_chunk for '
                f'full data shape {obj._res_shape} all on one process. If you '
                'see memory errors, try setting the col_chunk input to '
                'distribute the problem across multiple small workers.'
            )
        else:
            columns = np.arange(obj._res_shape[1])
            N = np.ceil(len(columns) / col_chunk)
            arrays = np.array_split(columns, N)
            slices = [slice(a[0], 1 + a[-1]) for a in arrays]
            logger.info(
                'MLClouds gap fill will be run across the full data '
                f'column shape {len(columns)} in {len(slices)} column chunks '
                f'with approximately {col_chunk} sites per chunk.'
            )

        if max_workers == 1:
            for col_slice in slices:
                out = obj.prep_chunk(
                    h5_source,
                    model_path=model_path,
                    var_meta=var_meta,
                    sza_lim=sza_lim,
                    col_slice=col_slice,
                )
                i_features, i_clean, i_flag = out
                out = obj.process_chunk(
                    i_features,
                    i_clean,
                    i_flag,
                    col_slice,
                    clean_data,
                    fill_flag,
                    low_mem=low_mem,
                )
                clean_data, fill_flag = out
                del i_features, i_clean, i_flag

        else:
            futures = {}
            logger.info(
                'Starting process pool for parallel phygnn cloud '
                f'fill with {max_workers} workers.'
            )
            loggers = ['nsrdb', 'rex', 'phygnn']
            with SpawnProcessPool(
                loggers=loggers, max_workers=max_workers
            ) as exe:
                for col_slice in slices:
                    future = exe.submit(
                        obj.prep_chunk,
                        h5_source,
                        model_path=model_path,
                        var_meta=var_meta,
                        sza_lim=sza_lim,
                        col_slice=col_slice,
                    )
                    futures[future] = col_slice

                logger.info(f'Kicked off {len(futures)} futures.')
                for i, future in enumerate(as_completed(futures)):
                    col_slice = futures[future]
                    i_features, i_clean, i_flag = future.result()
                    out = obj.process_chunk(
                        i_features,
                        i_clean,
                        i_flag,
                        col_slice,
                        clean_data,
                        fill_flag,
                        low_mem=low_mem,
                    )
                    clean_data, fill_flag = out
                    del i_features, i_clean, i_flag, future

                    mem = psutil.virtual_memory()
                    logger.info(
                        'MLCloudsFill futures completed: '
                        '{} out of {}. '
                        'Current memory usage is '
                        '{:.3f} GB out of {:.3f} GB total.'.format(
                            i + 1,
                            len(futures),
                            mem.used / 1e9,
                            mem.total / 1e9,
                        )
                    )

        obj.write_filled_data(clean_data, col_slice=slice(None))
        obj.write_fill_flag(fill_flag, col_slice=slice(None))
        obj.mark_complete_archived_files()
        logger.info(
            'Completed MLCloudsFill for h5_source: {}'.format(h5_source)
        )

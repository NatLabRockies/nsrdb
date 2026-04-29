"""Shared utilities for converting satellite data to UWISC format."""

import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import cached_property
from glob import glob

import numpy as np
import pandas as pd

DROP_VARS = ['relative_time']

UWISC_CLOUD_TYPE = {
    'N/A': -15,
    'Clear': 0,
    'Probably Clear': 1,
    'Fog': 2,
    'Water': 3,
    'Super-Cooled Water': 4,
    'Mixed': 5,
    'Opaque Ice': 6,
    'Cirrus': 7,
    'Overlapping': 8,
    'Overshooting': 9,
    'Unknown': 10,
    'Dust': 11,
    'Smoke': 12,
}


def expand_input_patterns(input_pattern):
    """Expand one or more glob patterns into a de-duplicated file list."""
    patterns = (
        [input_pattern]
        if isinstance(input_pattern, str)
        else list(input_pattern)
    )
    files = []
    for pattern in patterns:
        files.extend(glob(pattern, recursive=True))
    return list(dict.fromkeys(files))


def run_data_model_jobs(
    data_model_class,
    input_pattern,
    output_pattern,
    *,
    max_workers=None,
    group_inputs=None,
    logger=None,
):
    """Run a data-model conversion job over expanded input patterns."""
    logger = logger or logging.getLogger(data_model_class.__module__)
    files = expand_input_patterns(input_pattern)
    job_inputs = group_inputs(files) if group_inputs is not None else files

    if max_workers == 1:
        for input_data in job_inputs:
            data_model_class.run(input_data, output_pattern)
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for input_data in job_inputs:
                future = executor.submit(
                    data_model_class.run,
                    input_data,
                    output_pattern,
                )
                futures[future] = input_data

            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as error:
                    logger.error(
                        'Error processing file: %s',
                        futures[future],
                    )
                    logger.exception(error)

    logger.info('Finished converting %s files.', len(files))


class BaseUwiscDataModel:
    """Shared preprocessing pipeline for UWISC-format conversion."""

    NAME_MAP = {}
    CLOUD_TYPE_MAP = {}
    CLOUD_TYPE_SOURCE_VAR = None
    TIMESTAMP_PATTERN = r'.*_([0-9]+).([0-9]+).([0-9]+).\w+'

    def __init__(self, input_data, output_pattern):
        self.input_data = input_data
        self.output_pattern = output_pattern

    @staticmethod
    def _as_list(input_data):
        """Normalize a single input or grouped inputs into a list."""
        if isinstance(input_data, (list, tuple)):
            return list(input_data)
        return [input_data]

    @cached_property
    def input_files(self):
        """Get the normalized list of input files."""
        return self._as_list(self.input_data)

    @classmethod
    def get_primary_input_file(cls, input_files):
        """Get the primary input file used for naming outputs."""
        return input_files[0]

    @cached_property
    def primary_input_file(self):
        """Get the primary input file used for naming outputs."""
        return self.get_primary_input_file(self.input_files)

    @classmethod
    def parse_timestamp(cls, input_file):
        """Parse a timestamp tuple from an input file path."""
        ts = re.match(cls.TIMESTAMP_PATTERN, input_file).groups()
        if len(ts) == 3:
            year, doy, hour = ts
            return year, doy, hour, '000'
        return ts

    @classmethod
    def parse_timestamp_string(cls, input_file):
        """Parse the output timestamp string from an input file path."""
        return f's{"".join(cls.parse_timestamp(input_file))}'

    @cached_property
    def timestamp(self):
        """Get the parsed timestamp tuple for the primary input file."""
        return self.parse_timestamp(self.primary_input_file)

    @cached_property
    def timestamp_string(self):
        """Get the parsed output timestamp string for the primary input."""
        return self.parse_timestamp_string(self.primary_input_file)

    @cached_property
    def timeindex(self):
        """Get a single-step time index for the primary input file."""
        year, doy, hour, minute = self.timestamp
        timestamp = pd.to_datetime(
            f'{year}{doy}{hour}{minute}',
            format='%Y%j%H%M',
        )
        return pd.DatetimeIndex([timestamp])

    @cached_property
    def output_file(self):
        """Get output file name for the configured output pattern."""
        year, doy, _, _ = self.timestamp
        return self.output_pattern.format(
            year=year,
            doy=doy,
            timestamp=self.timestamp_string,
        )

    @classmethod
    def open_dataset(cls, input_data):
        """Open the raw input data as an xarray dataset."""
        raise NotImplementedError

    @cached_property
    def ds(self):
        """Get xarray dataset for raw input data."""
        return self.open_dataset(self.input_data)

    @classmethod
    def transform_raw_data(cls, ds):
        """Apply any subclass-specific preprocessing to the raw dataset."""
        return ds

    @classmethod
    def rename_vars(cls, ds):
        """Rename variables to uwisc conventions."""
        for current_name, target_name in cls.NAME_MAP.items():
            if current_name in ds.data_vars:
                ds = ds.rename({current_name: target_name})
        return ds

    @classmethod
    def drop_vars(cls, ds):
        """Drop variables that are not part of the UWISC output schema."""
        for var_name in DROP_VARS:
            if var_name in ds.data_vars:
                ds = ds.drop_vars(var_name)
        return ds

    @staticmethod
    def _rename_spatial_dims(ds):
        """Rename alternative spatial dimensions to the shared convention."""
        rename_map = {}
        if 'Lines' in ds.dims:
            rename_map['Lines'] = 'south_north'
        if 'Pixels' in ds.dims:
            rename_map['Pixels'] = 'west_east'
        if 'dim_y' in ds.dims:
            rename_map['dim_y'] = 'south_north'
        if 'dim_x' in ds.dims:
            rename_map['dim_x'] = 'west_east'
        if rename_map:
            ds = ds.rename(rename_map)
        return ds

    @staticmethod
    def _promote_lat_lon_coords(ds):
        """Ensure latitude and longitude are stored as coordinates."""
        sdims = ('south_north', 'west_east')
        if ('lat' in ds.coords or 'lat' in ds.data_vars) and (
            'latitude' not in ds.coords and 'latitude' not in ds.data_vars
        ):
            ds = ds.rename({'lat': 'latitude', 'lon': 'longitude'})

        if ds['latitude'].ndim == 1 and ds['longitude'].ndim == 1:
            ds['south_north'] = ds['latitude']
            ds['west_east'] = ds['longitude']

            lons, lats = np.meshgrid(ds['longitude'], ds['latitude'])
            ds = ds.assign_coords({
                'latitude': (sdims, lats),
                'longitude': (sdims, lons),
            })

        coord_names = [
            name for name in ('latitude', 'longitude') if name in ds.data_vars
        ]
        if coord_names:
            ds = ds.set_coords(coord_names)
        return ds

    def remap_dims(self, ds):
        """Rename dims and coords to standards and build 2D lat/lon grids."""
        for var_name in ds.data_vars:
            single_ts = (
                'time' in ds[var_name].dims
                and ds[var_name].transpose('time', ...).shape[0] == 1
            )
            if single_ts and var_name != 'reference_time':
                ds[var_name] = (
                    ('south_north', 'west_east'),
                    ds[var_name].isel(time=0).data,
                )

        ref_time = ds.attrs.get('reference_time', None)
        if ref_time is not None:
            time_index = pd.DatetimeIndex([ref_time]).values
        else:
            time_index = self.timeindex.values
        ds = self._rename_spatial_dims(ds)
        ds = self._promote_lat_lon_coords(ds)
        ds = ds.assign_coords({'time': ('time', time_index)})
        return ds

    @classmethod
    def remap_cloud_phase(cls, ds):
        """Map source cloud phase flags to UWISC cloud types."""
        if cls.CLOUD_TYPE_SOURCE_VAR is None:
            return ds

        cloud_type_name = cls.NAME_MAP[cls.CLOUD_TYPE_SOURCE_VAR]
        cloud_type = ds[cloud_type_name].values.copy()
        for value, cloud_source in cls.CLOUD_TYPE_MAP.items():
            cloud_type = np.where(
                ds[cloud_type_name].values.astype(int) == int(value),
                UWISC_CLOUD_TYPE[cloud_source],
                cloud_type,
            )
        ds[cloud_type_name] = (ds[cloud_type_name].dims, cloud_type)
        return ds

    @classmethod
    def derive_stdevs(cls, ds):
        """Derive standard deviations used as training features."""
        for var_name in ('refl_0_65um_nom', 'temp_11_0um_nom'):
            stddev = (
                ds[var_name]
                .rolling(
                    south_north=3,
                    west_east=3,
                    center=True,
                    min_periods=1,
                )
                .std()
            )
            ds[f'{var_name}_stddev_3x3'] = stddev
        return ds

    def process_dataset(self, ds):
        """Run the shared UWISC preprocessing pipeline on a dataset."""
        ds = self.remap_dims(ds)
        ds = self.transform_raw_data(ds)
        ds = self.rename_vars(ds)
        ds = self.drop_vars(ds)
        ds = self.remap_cloud_phase(ds)
        ds = self.derive_stdevs(ds)
        return ds

    @classmethod
    def write_output(cls, ds, output_file):
        """Write converted dataset to the final output file."""
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        ds = ds.transpose('south_north', 'west_east', ...)
        ds.load().to_netcdf(output_file, format='NETCDF4', engine='h5netcdf')

    @classmethod
    def run(cls, input_data, output_pattern):
        """Run the conversion routine and write the converted dataset."""
        logger = logging.getLogger(cls.__module__)
        data_model = cls(input_data, output_pattern)

        if os.path.exists(data_model.output_file):
            logger.info(
                '%s already exists. Skipping conversion.',
                data_model.output_file,
            )
            return

        logger.info('Geting xarray dataset for %s', input_data)
        ds = data_model.process_dataset(data_model.ds)

        logger.info('Writing converted file to %s', data_model.output_file)
        cls.write_output(ds, data_model.output_file)

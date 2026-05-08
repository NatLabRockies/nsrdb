"""Convert GK2A data to UWISC format."""

import argparse
import logging
import os
import re
from contextlib import suppress
from glob import glob

import numpy as np
import pandas as pd
import xarray as xr
from rex import init_logger

from nsrdb.preprocessing.base_data_model import (
    BaseUwiscDataModel,
    run_data_model_jobs,
)

init_logger('nsrdb', log_level='DEBUG')
init_logger(__name__, log_level='DEBUG')

logger = logging.getLogger(__name__)

NAME_MAP = {
    'sw038': 'temp_3_75um_nom',  # brightness temperature at 3.75 um (K)
    'ir112': 'temp_11_0um_nom',  # brightness temperature at 11.0 um (K)
    'vi006': 'refl_0_65um_nom',  # visible reflectance at 0.65 um (%)
    'COT': 'cld_opd_dcomp',  # cloud optical thickness
    'CER': 'cld_reff_dcomp',  # cloud effective radius
    'CTP': 'cld_press_acha',  # cloud top pressure
    'CTH': 'cld_height_acha',  # cloud top height
    'CP': 'cloud_type',  # cloud phase
    'VZA': 'sensor_zenith_angle',
    'VAZ': 'sensor_azimuth_angle',
}

GK2A_CLOUD_TYPE = {
    'Clear': 0,
    'Water phase': 1,
    'Ice phase': 2,
    'Uncertain phase': 6,
}

CLOUD_TYPE_MAP = {
    0: 'Clear',
    1: 'Water',
    2: 'Opaque Ice',
    6: 'Unknown',
}

# from https://nmsc.kma.go.kr/upload/resource/data/gk2a/20190415_GK-2A_AMI_Conversion_Table_v3.0.zip
VAR_CONSTANTS = {
    'vi006': {
        'gain': 0.154856294393539,
        'offset': -6.194244384765620,
        'cprime': 0.0019244840,
    },
    'sw038': {
        'gain': -0.00108296517282724000,
        'offset': 17.69998741149900000000,
        'center_wn': 2612.677373521110,
        'c0': -0.447843939824124,
        'c1': 1.000655680903890,
        'c2': -6.338240899124480e-08,
    },
    'ir112': {
        'gain': -0.02167448587715620000,
        'offset': 176.71343994140600000000,
        'center_wn': 891.713057301260,
        'c0': -0.249111718496148,
        'c1': 1.001211668737560,
        'c2': -1.131679640116650e-06,
    },
}


class Gk2aDataModel(BaseUwiscDataModel):
    """Class to handle conversion of gk2a data to standard uwisc style format
    for NSRDB pipeline"""

    NAME_MAP = NAME_MAP
    CLOUD_TYPE_MAP = CLOUD_TYPE_MAP
    CLOUD_TYPE_SOURCE_VAR = 'CP'

    @staticmethod
    def count_to_rad(ds, var, gain, offset):
        """Convert raw counts to radiance using gain and offset."""
        return ds[var] * gain + offset

    @classmethod
    def count_to_refl(cls, ds, var, gain, offset, cprime):
        """Convert raw counts to reflectance / albedo percent."""
        rad = cls.count_to_rad(ds, var, gain, offset)
        albedo = rad * cprime * 100
        return albedo

    @staticmethod
    def rad_to_temp(rad, center_wn):
        """Convert radiance to brightness temperature using Planck's law."""
        h = 6.62607015e-34
        c = 2.99792458e8
        k = 1.380649e-23

        wn_m = 100 * center_wn
        temp = (h * c) / (
            wn_m * k * np.log((2 * h * c**2 * wn_m**3) / (rad * 1e-5) + 1)
        )
        return temp

    @classmethod
    def count_to_temp(cls, ds, var, *, center_wn, gain, offset, c0, c1, c2):
        """Convert raw counts to brightness temperature."""
        rad = cls.count_to_rad(ds, var, gain, offset)
        te = cls.rad_to_temp(rad, center_wn)
        tb = c0 + c1 * te + c2 * te**2
        return tb

    @classmethod
    def transform_raw_data(cls, ds):
        """Convert raw counts to radiance for IR and visible channels."""
        temp_vars = (
            var for var in NAME_MAP if NAME_MAP[var].startswith('temp')
        )
        refl_vars = (
            var for var in NAME_MAP if NAME_MAP[var].startswith('refl')
        )
        for var in temp_vars:
            constants = VAR_CONSTANTS[var]
            ds[var] = cls.count_to_temp(ds, var, **constants)
        for var in refl_vars:
            constants = VAR_CONSTANTS[var]
            ds[var] = cls.count_to_refl(ds, var, **constants)
        return ds

    @classmethod
    def get_primary_input_file(cls, input_files):
        """Get the timestamped input file used for naming outputs."""
        for file in input_files:
            if re.search(r'(\d{12})(?=\.[^.]+$)', os.path.basename(file)):
                return file
        return input_files[0]

    @staticmethod
    def _normalize_channel_name(channel_name):
        """Normalize a channel name attribute to a lowercase variable name."""
        if isinstance(channel_name, (list, tuple, np.ndarray)):
            channel_name = channel_name[0]
        if hasattr(channel_name, 'item') and not isinstance(channel_name, str):
            with suppress(ValueError):
                channel_name = channel_name.item()
        if isinstance(channel_name, bytes):
            channel_name = channel_name.decode()
        return str(channel_name).lower()

    @classmethod
    def _rename_image_variable(cls, ds):
        """Rename image pixel values using the embedded channel name attr."""
        if 'image_pixel_values' not in ds.data_vars:
            return ds

        channel_name = ds['image_pixel_values'].attrs.get('channel_name')
        if channel_name is None:
            msg = 'image_pixel_values variable is missing channel_name attr'
            raise KeyError(msg)

        return ds.rename({
            'image_pixel_values': cls._normalize_channel_name(channel_name)
        })

    @staticmethod
    def _normalize_spatial_dims(ds):
        """Rename equivalent spatial dims so datasets can be merged."""
        rename_map = {}
        if 'x' in ds.dims:
            rename_map['x'] = 'dim_x'
        if 'y' in ds.dims:
            rename_map['y'] = 'dim_y'
        if rename_map:
            ds = ds.rename(rename_map)
        return ds

    @staticmethod
    def _get_spatial_shape(ds):
        """Get the standard spatial shape for a dataset if present."""
        if 'dim_y' in ds.dims and 'dim_x' in ds.dims:
            return ds.sizes['dim_y'], ds.sizes['dim_x']
        return None

    @classmethod
    def _coarsen_highres_dataset(cls, ds, target_shape):
        """Coarsen 4x higher-resolution datasets down to the target shape."""
        shape = cls._get_spatial_shape(ds)
        if shape is None or shape == target_shape:
            return ds

        y_size, x_size = shape
        target_y, target_x = target_shape
        y_factor = y_size // target_y
        x_factor = x_size // target_x
        if (
            y_size == target_y * 4
            and x_size == target_x * 4
            and y_factor == 4
            and x_factor == 4
        ):
            return ds.coarsen(dim_y=4, dim_x=4, boundary='trim').mean(
                keep_attrs=True
            )

        msg = (
            'Cannot align dataset with spatial shape '
            f'{shape} to target shape {target_shape}'
        )
        raise ValueError(msg)

    @classmethod
    def parse_timestamp(cls, input_file):
        """Parse the GK2A timestamp tuple from an input file path."""
        basename = os.path.basename(input_file)
        match = re.search(r'(\d{12})(?=\.[^.]+$)', basename)
        timestamp = pd.to_datetime(match.group(1), format='%Y%m%d%H%M')
        year = timestamp.strftime('%Y')
        doy = timestamp.strftime('%j')
        hour = timestamp.strftime('%H')
        minute = timestamp.strftime('%M')
        secs = '000'
        return year, doy, hour, minute, secs

    @classmethod
    def open_dataset(cls, input_files):
        """Get xarray dataset for raw input file"""
        return cls.combine_files(input_files)

    @classmethod
    def get_files_from_timestamp(cls, timestamp):
        """Get list of files needed for given timestamp. This is needed to
        combine different channels, which are stored in separate files."""
        year, doy, hour, minute, _ = timestamp
        file_pattern = f'*{year}.{doy}.{hour}{minute}*.nc'
        files = glob(os.path.join(os.path.dirname(__file__), file_pattern))
        return files

    @classmethod
    def combine_files(cls, files):
        """Combine multiple files into one dataset. This is needed to combine
        different channels, which are stored in separate files."""
        ds_list = []
        for file in files:
            ds = xr.open_dataset(file, format='NETCDF4', engine='h5netcdf')
            ds = cls._rename_image_variable(ds)
            ds = cls._normalize_spatial_dims(ds)
            ds_list.append(ds)

        spatial_shapes = [
            shape for ds in ds_list if (shape := cls._get_spatial_shape(ds))
        ]
        if spatial_shapes:
            target_shape = min(spatial_shapes)
            ds_list = [
                cls._coarsen_highres_dataset(ds, target_shape)
                for ds in ds_list
            ]

        return xr.merge(
            ds_list,
            compat='override',
            combine_attrs='drop_conflicts',
        )

    @classmethod
    def run(cls, input_files, output_pattern):
        """Run conversion routine and write converted dataset."""
        return super().run(input_files, output_pattern)


def group_files_by_timestamp(files):
    """Group files by timestamp, which is needed to combine different channels,
    which are stored in separate files."""
    groups = {}
    untimestamped = []
    for file in files:
        match = re.search(r'(\d{12})(?=\.[^.]+$)', os.path.basename(file))
        if match is None:
            untimestamped.append(file)
            continue

        timestamp = f's{match.group(1)}'
        if timestamp not in groups:
            groups[timestamp] = []
        groups[timestamp].append(file)

    if not groups:
        return [untimestamped.copy()] if untimestamped else []

    return [group + untimestamped for group in groups.values()]


if __name__ == '__main__':
    default_output_pattern = '/projects/pxs/GK2A/standardized/{year}'
    default_output_pattern += '/{doy}/gk2a_{timestamp}.nc'
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'input_pattern',
        type=str,
        nargs='+',
        help="""File pattern for input_files. e.g.
             /projects/pxs/GK2A/2025/**/*.nc""",
    )
    parser.add_argument(
        '-output_pattern',
        type=str,
        default=default_output_pattern,
        help='File pattern for output files.',
    )
    parser.add_argument(
        '-max_workers',
        type=int,
        default=10,
        help='Number of workers to use for parallel file conversion',
    )
    args = parser.parse_args()
    run_data_model_jobs(
        Gk2aDataModel,
        args.input_pattern,
        args.output_pattern,
        max_workers=args.max_workers,
        group_inputs=group_files_by_timestamp,
        logger=logger,
    )

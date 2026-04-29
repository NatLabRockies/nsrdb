"""Convert NASA data to UWISC format."""

import argparse
import logging

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
    'BT_3.75um': 'temp_3_75um_nom',
    'BT_10.8um': 'temp_11_0um_nom',
    'ref_0.63um': 'refl_0_65um_nom',
    'cloud_optical_depth': 'cld_opd_dcomp',
    'cloud_eff_particle_size': 'cld_reff_dcomp',
    'cloud_eff_pressure': 'cld_press_acha',
    'cloud_eff_height': 'cld_height_acha',
    'cloud_phase': 'cloud_type',
    'solar_zenith': 'solar_zenith_angle',
    'view_zenith': 'sensor_zenith_angle',
    'relative_azimuth': 'solar_azimuth_angle',
}

NASA_CLOUD_TYPE = {
    'Clear sky snow/ice': 0,
    'Water cloud': 1,
    'Ice cloud': 2,
    'No cloud property retrievals': 3,
    'Clear sky land/water': 4,
    'Bad input data': 5,
    'Possible water cloud': 6,
    'Possible ice cloud': 7,
    'Cleaned data': 13,
}

CLOUD_TYPE_MAP = {
    0: 'Clear',
    1: 'Water',
    2: 'Opaque Ice',
    3: 'Unknown',
    4: 'Clear',
    5: 'N/A',
    6: 'Water',
    7: 'Opaque Ice',
    13: 'Unknown',
}


class NasaDataModel(BaseUwiscDataModel):
    """Class to handle conversion of nasa data to standard uwisc style format
    for NSRDB pipeline"""

    NAME_MAP = NAME_MAP
    CLOUD_TYPE_MAP = CLOUD_TYPE_MAP
    CLOUD_TYPE_SOURCE_VAR = 'cloud_phase'

    @classmethod
    def open_dataset(cls, input_file):
        """Get xarray dataset for raw input file."""
        return xr.open_mfdataset(
            input_file,
            **{'group': 'map_data', 'decode_times': False},
            format='NETCDF4',
            engine='h5netcdf',
        )

    @classmethod
    def run(cls, input_file, output_pattern):
        """Run conversion routine and write converted dataset."""
        return super().run(input_file, output_pattern)


def run_jobs(input_pattern, output_pattern, max_workers=None):
    """Run multiple file conversion jobs."""
    return run_data_model_jobs(
        NasaDataModel,
        input_pattern,
        output_pattern,
        max_workers=max_workers,
        logger=logger,
    )


if __name__ == '__main__':
    default_output_pattern = '/projects/pxs/nasa_polar/standardized/{year}'
    default_output_pattern += '/{doy}/nacomposite_{timestamp}.nc'
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'input_pattern',
        type=str,
        nargs='+',
        help="""File pattern for input_files. e.g.
             /projects/pxs/nasa_polar/2023/*/*/*.nc""",
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
    run_jobs(
        input_pattern=args.input_pattern,
        output_pattern=args.output_pattern,
        max_workers=args.max_workers,
    )

"""CLI test for agg module."""

import json
import os
import shutil
import tempfile
import traceback

from rex import NSRDB

from nsrdb import TESTDATADIR, cli
from nsrdb.utilities.pytest import execute_pytest

meta_dir = os.path.join(TESTDATADIR, 'meta/')


IGNORE_DSETS = [
    'alpha',
    'asymmetry',
    'ssa',
    'ozone',
    'surface_albedo',
    'surface_pressure',
    'total_precipitable_water',
    'cloud_press_acha',
    'solar_zenith_angle',
    'clearsky_dni',
    'clearsky_dhi',
    'clearsky_ghi',
    'ghi',
    'dhi',
    'air_temperature',
    'dew_point',
    'relative_humidity',
    'wind_direction',
    'wind_speed',
    'wind_direction_10m',
    'wind_speed_10m',
    'cld_reff_dcomp',
]


def copy_dir(src, dst):
    """
    Copy all files in src to dst
    """
    if not os.path.exists(dst):
        os.makedirs(dst)

    for file in os.listdir(src):
        shutil.copy(os.path.join(src, file), dst)


def test_agg_cli(runner):
    """Make sure tmy runner functions correctly."""

    with tempfile.TemporaryDirectory() as td:
        fpath_multi = os.path.join(td, 'nsrdb_*_2018.h5')
        fpath_out = os.path.join(td, 'agg_out/agg_out_test_2018.h5')
        TESTJOB3 = {
            'source': {
                'fpath': fpath_multi,
                'tree_file': 'kdtree_surfrad_meta.pkl',
                'meta_file': 'surfrad_meta.csv',
                'spatial': '2km',
                'freq': '5min',
            },
            'final': {
                'fpath': fpath_out,
                'tree_file': 'kdtree_test_meta_agg.pkl',
                'meta_file': 'test_meta_agg.csv',
                'spatial': '4km',
                'freq': '30min',
            },
        }

        src = os.path.join(TESTDATADIR, 'nsrdb_2018')
        copy_dir(src, td)

        config = {
            'aggregate': {
                'data': TESTJOB3,
                'data_dir': td,
                'meta_dir': meta_dir,
                'n_chunks': 2,
                'year': 2018,
                'ignore_dsets': IGNORE_DSETS,
                'max_workers': 1,
            },
        }

        config_file = os.path.join(td, 'config.json')
        with open(config_file, 'w') as f:
            f.write(json.dumps(config))

        result = runner.invoke(cli.aggregate, ['-c', config_file])
        assert result.exit_code == 0, traceback.print_exception(
            *result.exc_info
        )

        fpath_out = fpath_out.replace('.h5', '_0.h5')
        with NSRDB(fpath_out, mode='r') as f:
            dsets = ('dni', 'aod', 'cloud_type', 'cld_opd_dcomp')
            assert all(d in f for d in dsets)

        fout = os.path.join(td, 'final_agg.h5')
        config = {
            'collect-aggregate': {
                'collect_dir': os.path.join(td, 'agg_out'),
                'meta_final': os.path.join(meta_dir, 'test_meta_agg.csv'),
                'collect_tag': 'agg_out_',
                'fout': fout,
                'max_workers': 1,
            },
        }

        config_file = os.path.join(td, 'config.json')
        with open(config_file, 'w') as f:
            f.write(json.dumps(config))

        result = runner.invoke(cli.collect_aggregate, ['-c', config_file])
        assert result.exit_code == 0, traceback.print_exception(
            *result.exc_info
        )

        with NSRDB(fout, mode='r') as f:
            dsets = ('dni', 'aod', 'cloud_type', 'cld_opd_dcomp')
            assert all(d in f for d in dsets)


if __name__ == '__main__':
    execute_pytest(__file__)

# pylint: skip-file
"""
PyTest file for PV generation in Rhode Island.

Created on Thu Nov 29 09:54:51 2018

@author: gbuster
"""

import datetime
import os
import tempfile

import h5py
import numpy as np
import pandas as pd
import pytest
from rex import Resource
from rex.utilities.loggers import init_logger

from nsrdb import DEFAULT_VAR_META, TESTDATADIR
from nsrdb.data_model import DataModel, VarFactory
from nsrdb.utilities.pytest import execute_pytest


def test_data_model_dump(var='asymmetry'):
    """Test dump routine with .tmp suffix"""
    with tempfile.TemporaryDirectory() as td:
        init_logger('nsrdb.data_model', log_file=None, log_level='DEBUG')

        out_file = os.path.join(td, var + '.h5')
        date = datetime.date(year=2017, month=1, day=1)
        var_meta = pd.read_csv(DEFAULT_VAR_META)
        var_meta['source_directory'] = os.path.join(TESTDATADIR, 'asymmetry')
        grid = os.path.join(
            TESTDATADIR, 'reference_grids/', 'west_psm_extent.csv'
        )
        data_model = DataModel(date, grid)
        data = data_model.run_single(var, date, grid, var_meta=var_meta)
        data = data_model.dump(var, out_file, data, purge=True, mode='w')

        assert os.path.exists(out_file)
        with Resource(out_file) as res:
            assert len(res.meta) == 1000
            assert 'gid' in res.meta
            assert len(res.time_index) == 288
            assert res['asymmetry'].shape == (288, 1000)


def test_asym(var='asymmetry'):
    """Test Asymmetry processed variables against baseline data."""
    init_logger('nsrdb.data_model', log_file=None, log_level='DEBUG')

    out_dir = os.path.join(TESTDATADIR, 'processed_ancillary/')
    date = datetime.date(year=2017, month=1, day=1)
    grid = os.path.join(TESTDATADIR, 'reference_grids/', 'west_psm_extent.csv')

    # set test directory
    var_meta = pd.read_csv(DEFAULT_VAR_META)
    var_meta['source_directory'] = os.path.join(TESTDATADIR, 'asymmetry')

    data = DataModel.run_single(var, date, grid, var_meta=var_meta)

    baseline_path = os.path.join(out_dir, var + '.h5')
    if not os.path.exists(baseline_path):
        with h5py.File(baseline_path, 'w') as f:
            f.create_dataset(var, data=data)
        msg = 'Output file for "{}" did not exist, created'.format(var)
        raise AssertionError(msg)
    with h5py.File(baseline_path, 'r') as f:
        data_baseline = f[var][...]
        var_obj = VarFactory.get_base_handler(
            var, var_meta=var_meta, date=date
        )
        data_baseline = var_obj.scale_data(data_baseline)
    assert np.allclose(data_baseline, data, atol=0, rtol=0.01)


@pytest.mark.parametrize(
    'var',
    (
        'surface_pressure',
        'air_temperature',
        'ozone',
        'total_precipitable_water',
        'wind_speed',
        'wind_speed_10m',
        'wind_direction',
        'wind_direction_10m',
        'specific_humidity',
        'alpha',
        'aod',
        'ssa',
        'solar_zenith_angle',
        'dew_point',
    ),
)
def test_ancillary_single(var):
    """Test MERRA processed variables against baseline data."""
    init_logger('nsrdb.data_model', log_file=None, log_level='DEBUG')

    out_dir = os.path.join(TESTDATADIR, 'processed_ancillary/')
    date = datetime.date(year=2017, month=1, day=1)
    grid = os.path.join(TESTDATADIR, 'reference_grids/', 'west_psm_extent.csv')

    # set test directory
    source_dir = os.path.join(TESTDATADIR, 'merra2_source_files/')
    var_meta = pd.read_csv(DEFAULT_VAR_META)
    var_meta['source_directory'] = source_dir

    # process integer-scaled data
    data = DataModel.run_single(
        var,
        date,
        grid,
        var_meta=var_meta,
        scale=True,
        factory_kwargs={'temporal_interp': 'linear'},
    )

    baseline_path = os.path.join(out_dir, var + '.h5')
    if not os.path.exists(baseline_path):
        with h5py.File(baseline_path, 'w') as f:
            f.create_dataset(var, data=data)
        msg = 'Output file for "{}" did not exist, created'.format(var)
        raise AssertionError(msg)
    with h5py.File(baseline_path, 'r') as f:
        data_baseline = f[var][...]

        # make sure baseline data is in integer precision
        var_obj = VarFactory.get_base_handler(
            var, var_meta=var_meta, date=date
        )
        data_baseline = var_obj.scale_data(data_baseline)

        # set data type to prevent overflow when doing error metrics
        data_baseline = data_baseline.astype(float)
        data = data.astype(float)

    bad = ~np.isclose(data_baseline, data, atol=1.0, rtol=0.0)
    diff = np.abs(data_baseline - data)
    rel_diff = np.abs(data_baseline - data) / data_baseline
    mean_baseline = np.mean(data_baseline)
    mean_test = np.mean(data)
    msg = (
        'Data for "{}" has {} values not close out of {} '
        'with abs diff of: max {:.3f}, mean {:.3f}, min {:.3f}, '
        'relative abs diff of: max {:.3f}, mean {:.3f}, min {:.3f} '
        'with a mean baseline data value of: {:.3f} and mean test data '
        'value of: {:.3f}. '
        '\nBad baseline values: {}, '
        '\nbad test values: {}'.format(
            var,
            bad.sum(),
            bad.shape[0] * bad.shape[1],
            diff.max(),
            diff.mean(),
            diff.min(),
            rel_diff.max(),
            rel_diff.mean(),
            rel_diff.min(),
            mean_baseline,
            mean_test,
            data_baseline[bad],
            data[bad],
        )
    )

    # abs tolerance has to be >=1 because we're comparing integer precision
    # still some small relative tolerance issues so also use rtol=1%
    # test only the data without the last interpolated time steps
    baseline_no_end = data_baseline[:-12]
    data_no_end = data[:-12]
    assert np.allclose(baseline_no_end, data_no_end, atol=1.0, rtol=0.01), msg
    # make sure the last interpolated time steps are within an acceptable
    # range but not constant
    baseline_end = data_baseline[-12:]
    data_end = data[-12:]
    assert not all(np.array_equal(x, data_end[0]) for x in data_end)
    assert np.allclose(
        [np.mean(data_end)], [np.mean(baseline_end)], atol=1.0, rtol=0.05
    )


def test_parallel(
    var_list=(
        'surface_pressure',
        'air_temperature',
        'specific_humidity',
        'relative_humidity',
    ),
):
    """Test the ancillary variable parallel processing with derived variable
    dependency."""
    init_logger('nsrdb.data_model', log_file=None, log_level='DEBUG')

    out_dir = os.path.join(TESTDATADIR, 'processed_ancillary/')
    date = datetime.date(year=2017, month=1, day=1)
    factory_kwargs = {
        v: {'source_dir': './'} for v in var_list if v in DataModel.CLOUD_VARS
    }
    grid = os.path.join(TESTDATADIR, 'reference_grids/', 'west_psm_extent.csv')

    # set test directory
    source_dir = os.path.join(TESTDATADIR, 'merra2_source_files/')
    var_meta = pd.read_csv(DEFAULT_VAR_META)
    var_meta['source_directory'] = source_dir

    data = DataModel.run_multiple(
        var_list,
        date,
        grid,
        var_meta=var_meta,
        max_workers=None,
        factory_kwargs=factory_kwargs,
    )

    for key, value in data.items():
        if key != 'time_index':
            baseline_path = os.path.join(out_dir, key + '.h5')
            if not os.path.exists(baseline_path):
                with h5py.File(baseline_path, 'w') as f:
                    f.create_dataset(key, data=value)
                msg = 'Output file for "{}" did not exist, created'.format(key)
                raise AssertionError(msg)
            with h5py.File(baseline_path, 'r') as f:
                data_baseline = f[key][...]
                var_obj = VarFactory.get_base_handler(
                    key, var_meta=var_meta, date=date
                )
                data_baseline = var_obj.scale_data(data_baseline)
            # comparing all but the last interpolated time steps
            assert np.allclose(
                data_baseline[:-12], value[:-12], atol=0.0, rtol=0.01
            )


@pytest.mark.parametrize(('interp_method', 'nn_method'), [('NN', 'haversine')])
def test_nrel_data_handler(nn_method, interp_method, var='aod'):
    """Test GFS processed variables"""
    date = datetime.date(year=2021, month=1, day=1)

    # NN to test GIDs 9 and 10
    grid = pd.DataFrame(
        {
            'latitude': [18.02, 18.18],
            'longitude': [-65.98, -65.98],
            'elevation': [0, 0],
        }
    )

    # set test directory
    source_dir = os.path.join(TESTDATADIR, 'clim_avg')
    factory_kwargs = {
        var: {
            'source_dir': source_dir,
            'handler': 'NrelVar',
            'file_set': 'pr_aod',
            'spatial_interp': interp_method,
            'nn_method': nn_method,
            'elevation_correct': False,
            'temporal_interp': 'linear',
        }
    }

    fp = os.path.join(source_dir, 'pr_aod.h5')
    with Resource(fp) as res:
        truth = res['aod', 0:48, 9:11]

    aod = DataModel.run_single(
        var,
        date,
        grid,
        factory_kwargs=factory_kwargs,
        scale=False,
        nsrdb_freq='30min',
    )
    assert np.allclose(aod, truth)


if __name__ == '__main__':
    execute_pytest(__file__)

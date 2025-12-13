"""
PyTest file for NSRDB file collection module

Created on Thu Nov 29 09:54:51 2018

@author: gbuster
"""

import ast
import itertools
import os
import shutil
import tempfile

import farms
import numpy as np
import pandas as pd
import pytest
import rest2
from pandas.testing import assert_frame_equal, assert_index_equal

import nsrdb
from nsrdb import TESTDATADIR
from nsrdb.data_model import VarFactory
from nsrdb.file_handlers.collection import Collector
from nsrdb.file_handlers.outputs import Outputs
from nsrdb.nsrdb import NSRDB
from nsrdb.utilities.file_utils import pd_date_range
from nsrdb.utilities.pytest import execute_pytest

RTOL = 0.05
ATOL = 0.1

BASELINE_4DAY = os.path.join(
    TESTDATADIR, 'collection_baseline/', 'collection_baseline_4day.h5'
)
BASELINE_ANNUAL = os.path.join(
    TESTDATADIR, 'collection_baseline/', 'collection_baseline_annual.h5'
)


def retrieve_data(fp, dset='air_temperature'):
    """Get data from h5."""

    with Outputs(fp, mode='r') as f:
        ti = f.time_index
        meta = f.meta
        data = f[dset]
        attrs = f.get_attrs(dset)

    return ti, meta, data, attrs


@pytest.mark.parametrize(
    ('sites', 'max_workers', 'n_writes', 'slim_meta'),
    (
        (None, 1, 1, False),
        (None, 2, 1, True),
        (None, 2, 2, True),
        (np.array([2, 3, 4]), 1, 1, True),
    ),
)
def test_collect_daily(sites, max_workers, n_writes, slim_meta):
    """Test chunk collection into daily file."""
    with tempfile.TemporaryDirectory() as td:
        src_dir = os.path.join(TESTDATADIR, 'data_model_daily_sample_output/')
        col_dir = os.path.join(td, 'collect_dir/')
        os.makedirs(col_dir)
        for fn in os.listdir(src_dir):
            col_fp = os.path.join(col_dir, fn)
            shutil.copy(os.path.join(src_dir, fn), col_fp)

            if slim_meta:
                with Outputs(col_fp, mode='a') as res:
                    meta_gids = res.meta[['gid']]
                    del res.h5['meta']
                    res.meta = meta_gids

        f_out = os.path.join(td, 'collected.h5')
        dsets = ['air_temperature', 'alpha']

        Collector.collect_daily(
            col_dir,
            f_out,
            dsets,
            sites=sites,
            max_workers=max_workers,
            n_writes=n_writes,
        )

        for dset in dsets:
            b_ti, b_meta, b_data, b_attrs = retrieve_data(
                BASELINE_4DAY, dset=dset
            )
            t_ti, t_meta, t_data, t_attrs = retrieve_data(f_out, dset=dset)

            if sites is not None:
                b_meta = b_meta.iloc[sites, :].reset_index(drop=True)
                b_data = b_data[:, sites]

            assert_index_equal(b_ti, t_ti)
            if slim_meta:
                assert_frame_equal(b_meta[['gid']], t_meta[['gid']])
            else:
                assert_frame_equal(b_meta, t_meta)

            assert np.allclose(b_data, t_data)
            for k, v in b_attrs.items():
                # source dirs have changed during re-orgs
                if k != 'source_dir':
                    assert str(v) == str(t_attrs[k])


def test_collect_lowmem():
    """Test a low memory file by file collection"""
    with tempfile.TemporaryDirectory() as td:
        flist = ['nsrdb_ancillary_2018_0.h5', 'nsrdb_ancillary_2018_1.h5']
        collect_dir = os.path.join(
            TESTDATADIR, 'data_model_annual_sample_output/'
        )
        f_out = os.path.join(td, 'collected.h5')

        dsets = ['surface_albedo', 'alpha', 'aod', 'surface_pressure']
        for dset in dsets:
            Collector.collect_flist_lowmem(flist, collect_dir, f_out, dset)

            ti, meta, data, attrs = retrieve_data(BASELINE_ANNUAL, dset=dset)

            with Outputs(f_out, mode='r') as f:
                attrs = f.get_attrs(dset=dset)
                meta_collected = f.meta
                data_collected = f[dset]

            assert 'scale_factor' in attrs
            assert 'data_source' in attrs
            assert 'spatial_interp_method' in attrs
            assert_frame_equal(meta, meta_collected)
            assert len(ti) == 105120

            # reduced precision for changed dtypes (aod in Jan 2022)
            msg = 'Dset failed: {}'.format(dset)
            assert np.allclose(data, data_collected, rtol=0.001, atol=0.001), (
                msg
            )


def test_final_daily():
    """Test the final collection of a daily data dir using synthetic test daily
    files."""
    with tempfile.TemporaryDirectory() as td:
        project_tdir = os.path.join(td, 'test_proj/')
        daily_dir = os.path.join(project_tdir, 'daily/')
        final_dir = os.path.join(project_tdir, 'final/')
        os.makedirs(daily_dir)
        grid_fp = os.path.join(
            TESTDATADIR, 'reference_grids/east_psm_extent.csv'
        )
        grid_df = pd.read_csv(grid_fp, index_col=0)
        grid_gids = pd.DataFrame({'gid': grid_df.index.values})
        shape = (24 * 12, len(grid_df))
        year = 2022
        doys = list(range(1, 42))

        test_data = {}
        for dsets in NSRDB.OUTS.values():
            for dset, doy in itertools.product(dsets, doys):
                test_data[dset] = test_data.get(dset, {})
                var_fac = VarFactory.get_base_handler(dset)
                low = var_fac.physical_min
                high = var_fac.physical_max
                dset_data = np.random.default_rng().uniform(
                    low, high, size=shape
                )
                dset_data = dset_data.astype(np.float32)
                test_data[dset][doy] = dset_data

                date_str0 = NSRDB.doy_to_datestr(year, doy)
                date_str1 = NSRDB.doy_to_datestr(year, doy + 1)
                ti = pd_date_range(
                    date_str0, date_str1, closed='left', freq='5min'
                )
                fn_out = '{}_{}_0.h5'.format(date_str0, dset)
                fp_out = os.path.join(daily_dir, fn_out)
                with Outputs(fp_out, 'w') as out:
                    out.meta = grid_gids
                    out.time_index = ti
                    out.write_dataset(dset, dset_data, dtype=np.float32)

        for i_fname in range(len(NSRDB.OUTS)):
            NSRDB.collect_data_model(
                project_tdir,
                year,
                grid_fp,
                n_chunks=1,
                i_chunk=0,
                i_fname=i_fname,
                freq='5min',
                max_workers=1,
                final_file_name='nsrdb',
                n_writes=2,
                final=True,
            )

        assert len(os.listdir(final_dir)) == 7
        fps = [os.path.join(final_dir, fn) for fn in os.listdir(final_dir)]
        for fp in fps:
            with Outputs(fp) as f:
                assert 'latitude' in f.meta
                assert 'longitude' in f.meta
                assert 'gid' not in f.meta
                assert len(f.meta) == len(grid_df)
                assert np.allclose(f.meta['latitude'], grid_df['latitude'])
                assert np.allclose(f.meta['longitude'], grid_df['longitude'])
                assert len(f.time_index) == 24 * 12 * len(doys)
                ti_doys = f.time_index.dayofyear
                assert np.allclose(sorted(ti_doys), ti_doys)
                assert all(d in ti_doys for d in doys)
                assert all(d in doys for d in ti_doys)

                dsets = [
                    d
                    for d in f.dsets
                    if d not in ('time_index', 'meta', 'coordinates')
                ]
                for dset, doy in itertools.product(dsets, doys):
                    disk_data = f[dset]
                    dset_test_data = test_data[dset][doy]
                    if np.issubdtype(disk_data.dtype, np.integer):
                        dset_test_data = np.round(dset_test_data)
                    mask = ti_doys == doy
                    disk_data_doy = disk_data[mask, :]
                    atol = 1 / f.attrs[dset]['scale_factor']
                    check = np.allclose(
                        disk_data_doy,
                        dset_test_data,
                        rtol=0.001,
                        atol=atol,
                    )
                    msg = '{} didnt match for doy {}'.format(dset, doy)
                    assert check, msg

                assert 'version_record' in f.global_attrs
                record = ast.literal_eval(f.global_attrs['version_record'])
                assert record['nsrdb'] == nsrdb.__version__
                assert record['farms'] == farms.__version__
                assert record['rest2'] == rest2.__version__


if __name__ == '__main__':
    execute_pytest(__file__)

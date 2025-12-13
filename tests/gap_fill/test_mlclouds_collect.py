"""
PyTest file for all sky daily processing after mlclouds daily gap fill

Created on 12/3/2020

@author: gbuster
"""

import os
import shutil
import tempfile

import numpy as np
import pytest
from rex import MultiFileNSRDB

from nsrdb import TESTDATADIR
from nsrdb.file_handlers.outputs import Outputs
from nsrdb.nsrdb import NSRDB
from nsrdb.utilities.pytest import execute_pytest

pytest.importorskip('mlclouds')
pytest.importorskip('phygnn')
from nsrdb.gap_fill.mlclouds_fill import MLCloudsFill

PROJECT_DIR = os.path.join(TESTDATADIR, 'mlclouds_pipeline/')
ARCHIVE_DIR = os.path.join(PROJECT_DIR, 'daily_files_archive/')
GRID = os.path.join(PROJECT_DIR, 'surfrad_meta.csv')


def test_collect(dates=('20190102', '20190103', '20190104'), slim_meta=True):
    """Test the mlclouds gap fill, allsky, and final collection."""
    with tempfile.TemporaryDirectory() as td:
        project_tdir = os.path.join(td, 'mlclouds_pipeline/')
        final_dir = os.path.join(project_tdir, 'final/')
        daily_dir = os.path.join(td, 'mlclouds_pipeline/daily/')
        shutil.copytree(ARCHIVE_DIR, daily_dir)

        if slim_meta:
            for fn in os.listdir(daily_dir):
                fp = os.path.join(daily_dir, fn)
                with Outputs(fp, mode='a') as res:
                    meta = res.meta[['gid', 'latitude', 'longitude']]
                    del res.h5['meta']
                    res.meta = meta

        year = int(dates[0][0:4])

        for date in dates:
            h5_source = os.path.join(daily_dir, '{}*.h5'.format(date))
            MLCloudsFill.run(h5_source)

            NSRDB.run_daily_all_sky(
                project_tdir, year, GRID, date, freq='5min', max_workers=1
            )

        for i_fname in range(len(NSRDB.OUTS)):
            NSRDB.collect_data_model(
                project_tdir,
                year,
                GRID,
                n_chunks=1,
                i_chunk=0,
                i_fname=i_fname,
                freq='5min',
                max_workers=2,
                final_file_name='mlclouds_test',
                n_writes=2,
                final=True,
            )

        fns = os.listdir(final_dir)
        assert len(fns) == 7
        assert all(fn.startswith('mlclouds_test_') for fn in fns)
        all_data = {}
        all_attrs = {}
        fp_final = os.path.join(final_dir, 'mlclouds_test_*.h5')
        with MultiFileNSRDB(fp_final) as res:
            dsets = res.dsets
            dsets = [
                d
                for d in dsets
                if d not in ('time_index', 'meta', 'coordinates')
            ]
            meta = res.meta
            ti = res.time_index
            assert len(dsets) == sum(len(v) for v in NSRDB.OUTS.values())
            assert len(meta) == 9
            assert len(ti) == (288 * 3)
            for dset in dsets:
                data = res[dset]
                assert (data != 0).sum() > 0, (
                    'Dataset {} is all zeros!'.format(dset)
                )
                all_data[dset] = data
                all_attrs[dset] = res.get_attrs(dset)

        with MultiFileNSRDB(h5_source) as res:
            L = len(res.time_index)
            for dset in dsets:
                assert np.allclose(
                    res[dset], all_data[dset][-L:, :], rtol=0.001, atol=0.001
                )

                attrs = res.get_attrs(dset)
                attrs_final = all_attrs[dset]

                assert 'units' in attrs
                assert 'scale_factor' in attrs
                assert 'psm_units' in attrs
                assert 'psm_scale_factor' in attrs

                for k, v in attrs.items():
                    assert k in attrs_final
                    if (
                        'dcomp' not in dset
                        and 'aod' not in dset
                        and k not in ('physical_max', 'source_dir')
                    ):
                        msg = '{}: {}: {} vs {}'.format(
                            dset, k, v, attrs_final[k]
                        )
                        assert str(v) == str(attrs_final[k]), msg


if __name__ == '__main__':
    execute_pytest(__file__)

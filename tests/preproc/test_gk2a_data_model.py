"""Tests for GK2A preprocessing helpers."""

from pathlib import Path

import numpy as np
import xarray as xr

from nsrdb.preprocessing.gk2a_data_model import (
    Gk2aDataModel,
    group_files_by_timestamp,
)


def test_group_files_by_timestamp_includes_untimestamped_files():
    """Untimestamped files should be included with each timestamp group."""
    files = [
        '/tmp/vi006_202501010700.nc',
        '/tmp/ir112_202501010700.nc',
        '/tmp/vi006_202501010710.nc',
        '/tmp/static_mask.nc',
    ]

    groups = group_files_by_timestamp(files)
    groups = sorted(sorted(group) for group in groups)

    assert groups == [
        sorted([
            '/tmp/ir112_202501010700.nc',
            '/tmp/static_mask.nc',
            '/tmp/vi006_202501010700.nc',
        ]),
        sorted([
            '/tmp/static_mask.nc',
            '/tmp/vi006_202501010710.nc',
        ]),
    ]


def test_group_files_by_timestamp_handles_only_untimestamped_files():
    """A list without timestamps should remain a single group."""
    files = ['/tmp/static_mask.nc', '/tmp/terrain.nc']

    assert group_files_by_timestamp(files) == [files]


def test_gk2a_output_filename_uses_year_doy_hour_minute_seconds():
    """GK2A output filenames should use NSRDB-style timestamp strings."""
    data_model = Gk2aDataModel(
        ['/tmp/vi006_202501010700.nc', '/tmp/static_mask.nc'],
        '/tmp/out/{year}/{doy}/gk2a_{timestamp}.nc',
    )

    assert data_model.timestamp_string == 's20250010700000'
    assert (
        data_model.output_file == '/tmp/out/2025/001/gk2a_s20250010700000.nc'
    )


def test_combine_files_coarsens_and_renames_image_variables(tmp_path):
    """Combine mixed-resolution files into one aligned dataset."""
    low_res = xr.Dataset({
        'COT': (('dim_y', 'dim_x'), np.array([[1.0, 2.0], [3.0, 4.0]]))
    })
    low_res_file = Path(tmp_path / 'cot_202501010700.nc')
    low_res.to_netcdf(low_res_file, engine='h5netcdf')

    high_res = xr.Dataset({
        'image_pixel_values': (
            ('y', 'x'),
            np.kron(
                np.array([[10.0, 20.0], [30.0, 40.0]]),
                np.ones((4, 4)),
            ),
        )
    })
    high_res['image_pixel_values'].attrs['channel_name'] = 'VI006'
    high_res_file = Path(tmp_path / 'vi006_202501010700.nc')
    high_res.to_netcdf(high_res_file, engine='h5netcdf')

    combined = Gk2aDataModel.combine_files([
        str(low_res_file),
        str(high_res_file),
    ])

    assert set(combined.data_vars) == {'COT', 'vi006'}
    assert combined['vi006'].dims == ('dim_y', 'dim_x')
    np.testing.assert_allclose(
        combined['vi006'].values,
        np.array([[10.0, 20.0], [30.0, 40.0]]),
    )


def test_gk2a_remap_cloud_phase_uses_cp_source_var():
    """GK2A cloud phase remapping should use the CP source variable mapping."""
    ds = xr.Dataset({
        'cloud_type': (
            ('dim_y', 'dim_x'),
            np.array([[0, 1], [2, 6]]),
        )
    })

    remapped = Gk2aDataModel.remap_cloud_phase(ds)

    np.testing.assert_array_equal(
        remapped['cloud_type'].values,
        np.array([[0, 3], [6, 10]]),
    )

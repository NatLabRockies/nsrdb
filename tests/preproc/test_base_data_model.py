"""Tests for shared preprocessing base model helpers."""

import numpy as np
import pytest
import xarray as xr

from nsrdb.preprocessing.base_data_model import (
    BaseUwiscDataModel,
    expand_input_patterns,
    run_data_model_jobs,
)


class DummyDataModel(BaseUwiscDataModel):
    """Minimal test double for the shared base preprocessing model."""


class DummyRunner:
    """Minimal runner target for shared run-data-model job tests."""

    calls = []

    @classmethod
    def run(cls, input_data, output_pattern):
        """Collect runner calls for assertions."""
        cls.calls.append((input_data, output_pattern))


@pytest.mark.parametrize(
    ('y_dim', 'x_dim'),
    [
        ('dim_y', 'dim_x'),
        ('Lines', 'Pixels'),
    ],
)
def test_remap_dims_normalizes_spatial_dims_and_coords(y_dim, x_dim):
    """remap_dims should normalize time, dims, and lat/lon coordinates."""
    model = DummyDataModel(
        '/tmp/input_2025.001.12.nc',
        '/tmp/out/{year}/{doy}/{timestamp}.nc',
    )
    ds = xr.Dataset(
        {
            'foo': (
                ('time', y_dim, x_dim),
                np.ones((1, 2, 2)),
            ),
            'latitude': (y_dim, np.array([10.0, 20.0])),
            'longitude': (x_dim, np.array([30.0, 40.0])),
        },
    )

    remapped = model.remap_dims(ds)

    assert remapped.indexes['time'][0] == model.time_index[0]
    assert remapped['foo'].dims == ('south_north', 'west_east')
    assert 'latitude' in remapped.coords
    assert 'longitude' in remapped.coords
    assert remapped['latitude'].dims == ('south_north', 'west_east')
    assert remapped['longitude'].dims == ('south_north', 'west_east')


def test_expand_input_patterns_handles_multiple_recursive_globs(
    make_nested_files,
):
    """Multiple recursive glob patterns should be expanded into one list."""
    file_1, file_2 = make_nested_files(
        'set_1/a/file_1.nc',
        'set_2/b/file_2.nc',
    )

    root = file_1.split('/set_1/', 1)[0]
    files = expand_input_patterns([
        f'{root}/set_1/**/*.nc',
        f'{root}/set_2/**/*.nc',
    ])

    assert files == [file_1, file_2]


def test_run_data_model_jobs_accepts_multiple_glob_patterns(
    make_nested_files,
):
    """Shared run_data_model_jobs should handle multiple glob inputs."""
    file_1, file_2 = make_nested_files(
        'set_1/a/file_1.nc',
        'set_2/b/file_2.nc',
    )
    DummyRunner.calls = []

    root = file_1.split('/set_1/', 1)[0]
    run_data_model_jobs(
        DummyRunner,
        [f'{root}/set_1/**/*.nc', f'{root}/set_2/**/*.nc'],
        '/tmp/out/{year}/{doy}/file_{timestamp}.nc',
        max_workers=1,
    )

    assert DummyRunner.calls == [
        (file_1, '/tmp/out/{year}/{doy}/file_{timestamp}.nc'),
        (file_2, '/tmp/out/{year}/{doy}/file_{timestamp}.nc'),
    ]

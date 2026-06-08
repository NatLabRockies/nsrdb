"""Shared pytest fixtures for preprocessing tests."""

import pytest


@pytest.fixture
def make_nested_files(tmp_path):
    """Create nested files beneath a temporary directory."""

    def _make(*relative_paths):
        files = []
        for relative_path in relative_paths:
            file_path = tmp_path / relative_path
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.touch()
            files.append(str(file_path))
        return files

    return _make


@pytest.fixture
def collect_run_calls(monkeypatch):
    """Patch a data model run method and collect its calls."""

    def _collect(data_model_class):
        calls = []

        def fake_run(input_data, output_pattern):
            calls.append((input_data, output_pattern))

        monkeypatch.setattr(data_model_class, 'run', fake_run)
        return calls

    return _collect

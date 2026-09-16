from pathlib import Path

import pytest

from download_data import EXPECTED_FILES, resolve_competition_path


def test_resolves_explicit_official_dataset(tmp_path: Path, monkeypatch) -> None:
    for filename in EXPECTED_FILES:
        (tmp_path / filename).touch()
    monkeypatch.setenv("WORCAP_DATA_DIR", str(tmp_path))

    assert resolve_competition_path() == tmp_path.resolve()


def test_rejects_incomplete_explicit_dataset(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "sample_submission.csv").touch()
    monkeypatch.setenv("WORCAP_DATA_DIR", str(tmp_path))

    with pytest.raises(FileNotFoundError, match="Diretório incompleto"):
        resolve_competition_path()

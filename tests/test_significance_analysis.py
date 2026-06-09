import os
from pathlib import Path

import pytest

from src.significance_analysis import find_latest_run_file


def test_find_latest_run_file_picks_latest(tmp_path: Path) -> None:
    method_dir = tmp_path / "base" / "scifact_2026-01-01_base"
    method_dir.mkdir(parents=True)
    older = method_dir / "scifact.run.trec"
    newer = method_dir / "nested" / "scifact.run.trec"
    newer.parent.mkdir()
    older.write_text("older")
    newer.write_text("newer")
    os.utime(older, (1_000_000, 1_000_000))
    os.utime(newer, (2_000_000, 2_000_000))

    result = find_latest_run_file("base", "scifact", tmp_path)
    assert result == newer


def test_find_latest_run_file_missing_method_dir(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Results directory not found"):
        find_latest_run_file("base", "scifact", tmp_path)


def test_find_latest_run_file_no_candidates(tmp_path: Path) -> None:
    (tmp_path / "base").mkdir()
    with pytest.raises(FileNotFoundError, match="No run file found"):
        find_latest_run_file("base", "scifact", tmp_path)

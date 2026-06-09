from pathlib import Path

import pytest

from src.download_data import download_datasets


def test_download_datasets_calls_beir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, str]] = []

    def fake_download(url: str, path: str) -> str:
        calls.append((url, path))
        dest = Path(path) / "scifact"
        dest.mkdir(parents=True)
        return str(dest)

    monkeypatch.setattr("src.download_data.util.download_and_unzip", fake_download)

    result = download_datasets(["scifact"], dest=tmp_path)

    assert result == [tmp_path / "scifact"]
    assert calls == [
        (
            "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip",
            str(tmp_path),
        )
    ]


def test_download_datasets_requires_names() -> None:
    with pytest.raises(ValueError, match="at least one dataset"):
        download_datasets([])

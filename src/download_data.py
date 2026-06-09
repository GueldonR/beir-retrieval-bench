"""Download BEIR datasets into DATA_ROOT/datasets/."""

import argparse
from pathlib import Path

from beir import util

from src.config.general_config import DATASET_STRINGS
from src.config.settings import DATA_ROOT

BEIR_URL = (
    "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip"
)


def download_datasets(
    datasets: list[str],
    dest: Path | None = None,
) -> list[Path]:
    if not datasets:
        raise ValueError("Provide at least one dataset name")

    output_dir = dest or (DATA_ROOT / "datasets")
    output_dir.mkdir(parents=True, exist_ok=True)

    downloaded: list[Path] = []
    for dataset in datasets:
        name = dataset.lower()
        url = BEIR_URL.format(name)
        print(f"Downloading {name}...")
        data_path = util.download_and_unzip(url, str(output_dir))
        path = Path(data_path)
        downloaded.append(path)
        print(f"Saved to {path}")

    return downloaded


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Download BEIR datasets into DATA_ROOT/datasets/",
    )
    parser.add_argument(
        "datasets",
        nargs="*",
        help=f"Dataset names (default: scifact). Known: {', '.join(DATASET_STRINGS)}",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all datasets listed in general_config.DATASET_STRINGS",
    )
    args = parser.parse_args(argv)

    if args.all:
        datasets = list(DATASET_STRINGS)
    elif args.datasets:
        datasets = args.datasets
    else:
        datasets = ["scifact"]

    download_datasets(datasets)


if __name__ == "__main__":
    main()

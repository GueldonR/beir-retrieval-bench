
from src.config.settings import DATA_ROOT

_DATASETS_DIR = DATA_ROOT / "datasets"
_RESULTS_DIR = DATA_ROOT / "results"
_SYNTHETIC_DIR = DATA_ROOT / "synthetic_data"

DATASET_PATHS = {
    dataset: _DATASETS_DIR / dataset / "corpus.jsonl"
    for dataset in ("fiqa", "quora", "scidocs", "scifact", "trec-covid")
}

DATASET_FOLDER_PATHS = {
    dataset: _DATASETS_DIR / dataset
    for dataset in ("fiqa", "quora", "scidocs", "scifact", "trec-covid")
}

RESULT_DIR = {
    "base": _RESULTS_DIR / "base",
    "hybrid": _RESULTS_DIR / "hybrid",
    "hype": _RESULTS_DIR / "hype",
    "hyde": _RESULTS_DIR / "hyde",
    "hybrid-rrf": _RESULTS_DIR / "hybrid-rrf",
}

RESULTS_ROOT = _RESULTS_DIR
PLOT_OUTPUT = _RESULTS_DIR / "forest_plot.png"

DATASET_STRINGS = ["fiqa", "quora", "scidocs", "scifact", "trec-covid"]

_HYDE_RUN_DIR = _SYNTHETIC_DIR / "hyde_documents" / "hyde_run_zero_temp"
_HYPE_RUN_DIR = _SYNTHETIC_DIR / "hype_queries" / "hype_run_temp_zero"

HYDE_DOCUMENTS = {
    dataset: _HYDE_RUN_DIR / f"{dataset}.hyde_documents.jsonl"
    for dataset in ("fiqa", "quora", "scidocs", "scifact", "trec-covid")
}

HYPE_QUERIES = {
    "fiqa": _HYPE_RUN_DIR / "fiqa.hype_queries.jsonl",
    "quora": _HYPE_RUN_DIR / "quora.hype_queries.jsonl",
    "scidocs": _HYPE_RUN_DIR / "scidocs.hype_queries.jsonl",
    "scifact": _HYPE_RUN_DIR / "scifact.hype_queries.jsonl",
    "trec-covid": _HYPE_RUN_DIR / "trec-covid.hype_queries.jsonl",
}

TARGET_DATASETS = ""

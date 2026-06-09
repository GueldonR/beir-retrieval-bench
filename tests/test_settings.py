from src.config.general_config import DATASET_FOLDER_PATHS, PLOT_OUTPUT, RESULTS_ROOT
from src.config.settings import DATA_ROOT, EMBEDDING_URL, QDRANT_URL, VLLM_URL


def test_data_root_is_absolute():
    assert DATA_ROOT.is_absolute()


def test_dataset_paths_under_data_root():
    for path in DATASET_FOLDER_PATHS.values():
        assert path.is_relative_to(DATA_ROOT)


def test_results_paths_under_data_root():
    assert RESULTS_ROOT.is_relative_to(DATA_ROOT)
    assert PLOT_OUTPUT.is_relative_to(DATA_ROOT)


def test_service_urls_have_defaults():
    assert QDRANT_URL.startswith("http")
    assert VLLM_URL.startswith("http")
    assert EMBEDDING_URL.startswith("http")

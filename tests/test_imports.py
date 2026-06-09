import importlib

import pytest

MODULES = [
    "src.config.settings",
    "src.config.general_config",
    "src.download_data",
    "src.components.db",
    "src.components.embedding",
    "src.components.llm",
    "src.beir_test",
    "src.setup_infra",
    "src.significance_analysis",
]


@pytest.mark.parametrize("module", MODULES)
def test_import_core_modules(module: str) -> None:
    importlib.import_module(module)

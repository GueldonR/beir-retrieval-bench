import os
from pathlib import Path

from dotenv import load_dotenv

_REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(_REPO_ROOT / ".env")


def _resolve_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (_REPO_ROOT / path).resolve()


DATA_ROOT = _resolve_path(os.getenv("DATA_ROOT", "./data"))
HF_CACHE_DIR = _resolve_path(os.getenv("HF_CACHE_DIR", str(DATA_ROOT / "hub")))

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
VLLM_URL = os.getenv("VLLM_URL", "http://localhost:8100/v1")
EMBEDDING_URL = os.getenv("EMBEDDING_URL", "http://localhost:12434/v1")
HF_TOKEN = os.getenv("HF_TOKEN", "")

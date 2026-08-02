# beir-retrieval-bench

Retrieval benchmark on [BEIR](https://github.com/beir-cellar/beir) datasets: **baseline** (dense), **HyDE**, **HyPE**, and **hybrid** (dense + BM25 via RRF). Stack: Qdrant, Qwen embeddings, vLLM.

**Needs:** Python 3.10+, [uv](https://docs.astral.sh/uv/), Docker, NVIDIA GPU + `HF_TOKEN` for full runs. CPU-only works for `significance_analysis.py` (no Qdrant or vLLM required).

## Quick start

```bash
cp .env.example .env          # set DATA_ROOT, HF_TOKEN
# PowerShell: Copy-Item .env.example .env
uv sync --group dev
docker compose up -d          # full stack (GPU); CPU-only: docker compose up -d qdrant
```

### Data layout

Benchmark data lives under `DATA_ROOT` (default `./data/` in the project folder). It is **not committed to git** — BEIR corpora are large (100 MB–several GB per dataset), so `data/` is in `.gitignore`. Each machine downloads its own copy on first run.

```
data/
├── datasets/          # BEIR downloads (corpus, queries, qrels)
├── synthetic_data/    # HyDE/HyPE LLM outputs (you generate or copy in)
├── results/           # evaluation .run.trec files + forest_plot.png
└── hub/               # Hugging Face model cache (Docker volume)
```

Point `DATA_ROOT` in `.env` elsewhere if you prefer (e.g. a shared lab drive).

### Run pipeline

All commands run from the repo root. Each entry script has an `if __name__ == "__main__"` block — **edit that block** to pick dataset, method, and collection name before running.

**1. Download a BEIR dataset** (once per machine):

```bash
uv run python -m src.download_data scifact
```

For all five benchmark datasets:

```bash
uv run python -m src.download_data --all
```

Files land under `./data/datasets/<name>/` from the public BEIR mirror.

**2. Ingest into Qdrant** (needs Qdrant + embedding service running):

```bash
uv run python -m src.setup_infra
```

Default `__main__` calls `setup_infra_HyPE()` for scifact. Switch to `setup_infra_base()`, `setup_infra_all()`, etc. as needed. HyPE/HyDE ingestion also needs matching files under `data/synthetic_data/` (paths in `src/config/general_config.py`).

**3. Evaluate** (needs Qdrant with ingested collection):

```bash
uv run python -m src.beir_test
```

Default runs `baseEvalutation("scifact", "scifact")`. Collection name must match what step 2 created (e.g. `"scifact_hype"` after `setup_infra_all_HyPE()`).

**4. Significance analysis** (CPU-only; no Docker required):

```bash
uv run python -m src.significance_analysis
```

Needs qrels from step 1 and `.run.trec` files from step 3 under `data/results/{base,hyde,hybrid-rrf,hype}/`. Run all four evaluation methods first, or copy existing run files in.

## Config

| Variable | Default |
|----------|---------|
| `DATA_ROOT` | `./data` |
| `HF_TOKEN` | — |
| `HF_CACHE_DIR` | `./data/hub` |
| `QDRANT_URL` | `http://localhost:6333` |
| `VLLM_URL` | `http://localhost:8100/v1` |
| `EMBEDDING_URL` | `http://localhost:12434/v1` |

GPU workloads run via Docker (`docker compose`); no local PyTorch dependency.

```bash
uv run pytest tests/ -q
```

## Layout

- `src/components/` — Qdrant, embeddings, LLM
- `src/logic/` — ingestion + evaluation per method
- `src/download_data.py`, `src/setup_infra.py`, `src/beir_test.py`, `src/significance_analysis.py` — entry points

## Troubleshooting

- **Qdrant connection refused** → `docker compose up -d qdrant`
- **No run files for significance analysis** → run evaluation first, or copy `.run.trec` files into `$DATA_ROOT/results/`
- **vLLM exits** → check `nvidia-smi` and `HF_TOKEN`
- **Dataset not found** → download BEIR data into `$DATA_ROOT/datasets/`
- **Import errors** → run from repo root with `uv run`

Windows GPU: use WSL2 + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

Datasets and evaluation tooling from [BEIR](https://github.com/beir-cellar/beir) (Thakur et al.).

MIT — see [LICENSE](LICENSE).

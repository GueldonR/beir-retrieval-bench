# Single source for setup_infra (ingest) and beir_test (eval)
DATASET = "scifact"

COLLECTIONS = {
    "base": DATASET,
    "hybrid": f"{DATASET}_hybrid",
    "hype": f"{DATASET}_hype",
}

EMBEDDING_BATCH_SIZE = 20

import json
import uuid
from typing import Optional

from qdrant_client.models import PointStruct
from typing_extensions import override

from src.config import HYPE_QUERIES
from src.logic.ingestion.IngestDataLogic import IngestDataLogic


class IngestDataLogicHype(IngestDataLogic):
    """
    Handles the ingestion flow for Hype indexing phase

    Uses pregenerated queries: "HYPE_QUERIES"
    """
    def __init__(self, target_dataset, target_collection, qdrant_instance=None):
        super().__init__(target_dataset, target_collection, qdrant_instance)
        if target_dataset not in HYPE_QUERIES:
            raise ValueError("Target dataset does not have a correctly formatted HyPE query file")
        self.hype_path = HYPE_QUERIES.get(target_dataset)
        self.hype_prompts = self._get_dict(self.hype_path)

    @override
    def ingest_data(self, limit_docs: Optional[int] = None, upload_batch_size: int = 250, embedding_batch_size: int = 32):
        points = []
        batch_docs = []
        doc_counter = 0

        for doc in self._iter_dataset(self.dataset):
            if not doc:
                continue

            if limit_docs and doc_counter >= limit_docs:
                print(f"Limit of {limit_docs} docs reached")
                break

            doc_counter += 1
            
            doc_id = str(doc.get("_id"))
            hypo_texts = self.hype_prompts.get(doc_id)

            if not hypo_texts:
                raise ValueError(f"Missing HyPE queries for doc_id {doc_id}")

            for hypo_text in hypo_texts:
            
                batch_docs.append({"doc_id": doc_id, "text": hypo_text})
                if len(batch_docs) >= embedding_batch_size:
                    points.extend(self._create_points_batch(batch_docs))
                    batch_docs.clear()
               
                if len(points) >= upload_batch_size:
                    print(f"Processed {doc_counter} embeddings. Uploading batch...")
                    self._upload_batch(points)
                    points.clear()

            if doc_counter % 100 == 0:
                print(f"Processed {doc_counter} documents...")

        if batch_docs:
            points.extend(self._create_points_batch(batch_docs))

        if points:
            print(f"Uploading final batch of {len(points)} hypo_texts...")
            self._upload_batch(points)

    @override
    def _create_points_batch(self, docs_batch: list[dict]):
        if not docs_batch:
            return []

        texts = [doc["text"] for doc in docs_batch]
        vectors_batch = self.embedding_model.embed_texts_batch(texts)

        points = []
        for i, (doc, vec) in enumerate(zip(docs_batch, vectors_batch)):
            points.append(PointStruct(
                id=str(uuid.uuid4()),
                vector=vec,
                payload={
                    "doc_id": doc["doc_id"],  # original document id
                    "text": doc["text"] # hypo text
                }
            ))
        return points

    def _get_dict(self, path):
        data = {}
        with open(path, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                data[rec["_id"]] = rec["hypothetical_queries"]  # list of hypo queries
        return data
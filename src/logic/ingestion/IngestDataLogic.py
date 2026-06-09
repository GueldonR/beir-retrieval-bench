import json
import uuid
from typing import Optional

from qdrant_client import models
from qdrant_client.models import PointStruct

from src.components.db import QdrantInstance
from src.components.embedding import EmbeddingModelInstance
from src.config import DATASET_PATHS


class IngestDataLogic:

    def __init__(self, target_dataset, target_collection, qdrant_instance = None, sparse_model_name="Qdrant/bm25", hybrid_search=False):
        self.qdrant_client = qdrant_instance or QdrantInstance()
        self.collection_name = target_collection
        self.embedding_model = EmbeddingModelInstance()
        self.sparse_model_name = sparse_model_name
        self.hybrid_search = hybrid_search
        self.dataset = DATASET_PATHS.get(target_dataset)
        if not self.dataset:
            raise ValueError(f"Dataset '{target_dataset}' not found in DATASET_PATHS")

    def ingest_data(self, limit_docs: Optional[int] = None, upload_batch_size: int = 250, embedding_batch_size: int = 64):
        points = []
        batch_docs = []
        doc_counter = 0
          
        for doc in self._iter_dataset(self.dataset):
            if not doc: 
                raise ValueError("doc returned empty line")

            if limit_docs and doc_counter >= limit_docs:
                break

            doc_counter += 1

            # collect documents into batch
            batch_docs.append(doc)
            # batch embeddings with the batch-doc starting id
            if len(batch_docs) >= embedding_batch_size:
                points.extend(self._create_points_batch(batch_docs))
                batch_docs.clear()

            
            if len(points) >= upload_batch_size:
                print(f"Processed {doc_counter} docs. Uploading batch...")
                self._upload_batch(points)
                points.clear()

            if doc_counter % 100 == 0:
                print(f"Processed {doc_counter} documents...")

        # flush remaining batch docs
        if batch_docs:
            points.extend(self._create_points_batch(batch_docs))

        # upload any remaining points
        if points:
            print(f"Uploading final batch of {len(points)} documents...")
            self._upload_batch(points)

        ### Close the client on the DB.py class

    """
    Helpers
    """
    def _iter_dataset(self, dataset_path):
        """
        Traverses singular datasets
        param - instance, a valid path
        yields - a line of the dataset
        """
        with open(dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                try: 
                    yield json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"Skipping malformed line: {e}")

    def _build_full_text(self, doc: dict) -> str:
        """
        Takes a dictonary and concatenates the field to make full text
        param - instance, dictionary 
        returns - concatenated string
        """
        if not doc:
            raise ValueError("Doc is missing")

        return doc.get("title", "") + "\n" + doc.get("text", "")

    def _upload_batch(self, points):
        """
        Uploads the points in a batch
        param - instance, list of points (batch)
        """
        try: 
            self.qdrant_client.client.upsert(
                collection_name=self.collection_name,
                points=points,
            )
        except Exception as e:
            print(f"Upsert failed: {e}")
            
        
    def _create_point(self, doc, doc_counter):
        """
        DEPRECATED!
        Embeds a single doc and returns a point
        param - instance, doc, count of processed docs
        returns - a pointstruct: internal (qdrant) id, vector and the payload
        NOTE: The internal id is not the same as the doc_id. The latter is searched when benchmarking.
        """
        if not doc:
            return None

        full_text = self._build_full_text(doc)
        vector = self.embedding_model.embed_text(full_text)

        if self.hybrid_search:
            point_vector = {
                "dense": vector,
                "sparse": models.Document(text=full_text, model=self.sparse_model_name),
            }
        else:
            point_vector = vector

        return PointStruct(
            id=doc_counter,
            vector=point_vector,
            payload={
                "doc_id": str(doc.get("_id")),
                "text": full_text
            },
        )
    
    def _create_points_batch(self, docs_batch: list[dict]):
        """
        Embeds a list of docs and maps them into a list of pointstructs
        param - instance, list of docs, the id from where to start for the batch
        returns - list of points
        """
        if not docs_batch:
            return []
        
        texts = [self._build_full_text(doc) for doc in docs_batch]

        vectors_batch = self.embedding_model.embed_texts_batch(texts)  
        
        points = []

        for i, (doc, vec) in enumerate(zip(docs_batch, vectors_batch)):
            if self.hybrid_search:
                point_vector = {
                    "dense": vec,
                    "sparse": models.Document(text=texts[i], model=self.sparse_model_name),
                }
            else:
                point_vector = vec

            points.append(PointStruct(
                id=str(uuid.uuid4()),
                vector=point_vector,
                payload={
                    "doc_id": str(doc.get("_id")),
                    "text": texts[i]
                }
            ))
        return points
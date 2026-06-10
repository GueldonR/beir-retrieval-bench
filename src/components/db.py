from qdrant_client import QdrantClient
from qdrant_client import models as qdrant_models
from qdrant_client.models import QueryRequest, SparseVectorParams, VectorParams

from src.config.settings import QDRANT_URL


class QdrantInstance:
    def __init__(self):
        self.client = QdrantClient(url=QDRANT_URL)
        self.dimensions = 1024

    def setup_collection(self, collection_name, hybrid_search=False):
        """
        Creates a collection if the collection_name does not exist
        """
        collections = self.client.get_collections()
        existing_collections = {c.name for c in collections.collections}

        if collection_name not in existing_collections:
            if hybrid_search:
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config={
                        "dense": VectorParams(
                            size=self.dimensions,
                            distance=qdrant_models.Distance.COSINE,
                        )
                    },
                    sparse_vectors_config={
                        "sparse": SparseVectorParams(
                            modifier=qdrant_models.Modifier.IDF,
                        )
                    },
                )
            else:
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=self.dimensions, distance=qdrant_models.Distance.COSINE
                    ),
                )
            print(f"Created collection: {collection_name}")
        else:
            print(f"Collection already exists: {collection_name}")

    def close_connection(self):
        """Closes the connection to the instansiated Qdrant object"""
        self.client.close()

    def delete_all_collection(self):

        collections = self.client.get_collections()

        for collection in collections.collections:
            self.client.delete_collection(collection.name)
            print(f"Deleted collection: {collection.name}")

        print("Database cleared!")

    def search_vector_space(self, target_collection_name, query, search_limit=10):
        """
        Takes a queryvector and returns a search result
        """
        search_result = self.client.query_points(
            collection_name=target_collection_name,
            query=query,
            with_payload=True,
            limit=search_limit,
        )
        return search_result.points

    def search_vector_space_batch(
        self, collection_name: str, queries: list, top_k: int = 10, with_payload: bool = True
    ):
        """
        Takes a list of queryvectors and returns a list of QueryResponse objects
        """
        requests = [QueryRequest(query=q, limit=top_k, with_payload=with_payload) for q in queries]
        return self.client.query_batch_points(collection_name=collection_name, requests=requests)

    def search_vector_space_batch_rrf(
        self,
        collection_name: str,
        query_texts: list[str],
        dense_queries: list,
        sparse_model: str = "Qdrant/bm25",
        top_k: int = 10,
        dense_prefetch: int = 100,
        sparse_prefetch: int = 100,
        with_payload: bool = True,
    ):
        requests = [
            QueryRequest(
                prefetch=[
                    qdrant_models.Prefetch(
                        query=qdrant_models.Document(text=text, model=sparse_model),
                        using="sparse",
                        limit=sparse_prefetch,
                    ),
                    qdrant_models.Prefetch(
                        query=dense,
                        using="dense",
                        limit=dense_prefetch,
                    ),
                ],
                query=qdrant_models.FusionQuery(fusion=qdrant_models.Fusion.RRF),
                limit=top_k,
                with_payload=with_payload,
            )
            for text, dense in zip(query_texts, dense_queries)
        ]
        return self.client.query_batch_points(collection_name=collection_name, requests=requests)

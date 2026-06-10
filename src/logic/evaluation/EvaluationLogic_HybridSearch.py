from typing_extensions import override

from src.logic.evaluation.EvaluationLogic import EvaluationLogic


class EvaluationLogic_HybridSearch(EvaluationLogic):
    def __init__(
        self,
        score_function="cos_sim",
        vector_store=None,
        embedding_model_instance=None,
        target_dataset="scifact",
    ):
        super().__init__(score_function, vector_store, embedding_model_instance, target_dataset)

    @override
    def run_evaluation(
        self, target_collection, dense_prefetch: int = 100, sparse_prefetch: int = 100
    ):

        queries, qrels, results, batch_size = self.get_run_configurations(batch_size=32)
        collection_name = target_collection

        query_ids = list(qrels.keys())
        query_texts = [queries[qid] for qid in query_ids]

        for i in range(0, len(query_texts), batch_size):
            batch_texts = query_texts[i : i + batch_size]
            batch_qids = query_ids[i : i + batch_size]

            batch_vecs = self.embedding_model.embed_texts_batch(batch_texts, is_query=True)

            batch_responses = self.vector_store.search_vector_space_batch_rrf(
                collection_name=collection_name,
                query_texts=batch_texts,
                dense_queries=batch_vecs,
                top_k=self.top_k_max,
                dense_prefetch=dense_prefetch,
                sparse_prefetch=sparse_prefetch,
                with_payload=True,
                sparse_model="Qdrant/bm25",
            )

            for qid, res in zip(batch_qids, batch_responses):
                if res is None:
                    raise ValueError(f"Missing response for qid {qid}")
                results[qid] = {
                    point.payload["doc_id"]: point.score for point in res.points if point.payload
                }

        ndcg, _map, recall, precision, mrr, per_query_scores = self._get_evaluation_result(
            qrels=qrels, results=results
        )

        self.save_results_to_instance(results, ndcg, _map, recall, precision, mrr, per_query_scores)

        self._result_printer(
            ndcg=self.ndcg,
            mrr=self.mrr,
            precision=self.precision,
            _map=self._map,
            recall=self.recall,
        )

        self._save_to_disk(test_type="hybrid-rrf")

        return results, ndcg, _map, recall, precision, mrr, per_query_scores

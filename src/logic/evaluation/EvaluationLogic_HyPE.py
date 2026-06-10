from typing_extensions import override

from src.config.general_config import HYPE_QUERIES
from src.logic.evaluation.EvaluationLogic import EvaluationLogic


class EvaluationLogic_HyPE(EvaluationLogic):
    def __init__(
        self,
        score_function="cos_sim",
        vector_store=None,
        embedding_model_instance=None,
        target_dataset="scifact",
    ):
        super().__init__(score_function, vector_store, embedding_model_instance, target_dataset)
        self.hype_prompts_path = HYPE_QUERIES[
            self.target_dataset_string
        ]  # target_dataset_string är en lowercase sträng
        self.top_k_max = 500

    @override
    def run_evaluation(self, target_collection):

        queries, qrels, results, batch_size = self.get_run_configurations(batch_size=20)
        collection_name = target_collection
        query_ids = list(qrels.keys())
        query_texts = [queries[qid] for qid in query_ids]

        for i in range(0, len(query_texts), batch_size):
            batch_texts = query_texts[i : i + batch_size]
            batch_qids = query_ids[i : i + batch_size]

            # embed batch vectors
            batch_vecs = self.embedding_model.embed_texts_batch(batch_texts, is_query=True)

            # Batch search request
            batch_responses = self.vector_store.search_vector_space_batch(
                collection_name=collection_name,
                queries=batch_vecs,
                top_k=self.top_k_max,
                with_payload=True,
            )

            for qid, res in zip(batch_qids, batch_responses):
                # removes and maxes out scores for duped ids
                results[qid] = self._aggregate_retrived_results_(qid=qid, res=res)

        ndcg, _map, recall, precision, mrr, per_query_scores = self._get_evaluation_result(
            qrels=qrels, results=results
        )

        # store metrics for _save_to_disk()
        self.save_results_to_instance(results, ndcg, _map, recall, precision, mrr, per_query_scores)

        self._result_printer(
            ndcg=self.ndcg,
            mrr=self.mrr,
            precision=self.precision,
            _map=self._map,
            recall=self.recall,
        )
        self._save_to_disk(test_type="hype")

        return results, ndcg, _map, recall, precision, mrr, per_query_scores

    def _aggregate_retrived_results_(self, qid: str, res):
        doc_scores = {}

        for point in res.points:
            doc_id = point.payload["doc_id"]
            score = point.score
            # Keep first score as-is; then keep max across duplicate HyPE variants per doc.
            if doc_id in doc_scores:
                doc_scores[doc_id] = max(doc_scores[doc_id], score)
            else:
                doc_scores[doc_id] = score

        ranked = dict(
            sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[: self.top_k_max]
        )

        return ranked

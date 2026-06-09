import json

from typing_extensions import override

from src.config import HYDE_DOCUMENTS
from src.logic.evaluation.EvaluationLogic import EvaluationLogic


class EvaluationLogic_HyDE(EvaluationLogic):

    def __init__(self, score_function="cos_sim", vector_store=None, embedding_model_instance=None, target_dataset="scifact"):
        super().__init__(score_function, vector_store, embedding_model_instance, target_dataset)
        if self.target_dataset_string not in HYDE_DOCUMENTS:
            raise ValueError("Dataset is not valid, check path or target dataset")
        self.hypo_queries_path = HYDE_DOCUMENTS[self.target_dataset_string] # target_dataset_string är en lowercase sträng

    @override
    def run_evaluation(self, target_collection):

            queries,qrels,results,batch_size = self.get_run_configurations()
            collection_name = target_collection            
            missing = set(qrels.keys()) - set(queries.keys())
            if missing:
                raise ValueError(f"Missing HyDE passages for {len(missing)} queries")

            query_ids = list(qrels.keys())
            query_texts = [queries[qid] for qid in query_ids]

            for i in range(0, len(query_texts), batch_size):
                batch_texts = query_texts[i:i+batch_size]
                batch_qids = query_ids[i:i+batch_size]

                # embed batch vectors 
                batch_vecs = self.embedding_model.embed_texts_batch(batch_texts, is_query=True)

                # Batch search request
                batch_responses = self.vector_store.search_vector_space_batch(
                collection_name=collection_name,
                queries=batch_vecs,
                top_k=self.top_k_max,
                with_payload=True
                )


                for qid, res in zip(batch_qids, batch_responses):
                    if res is None:
                        raise ValueError(f"Missing response for qid {qid}") 
                    results[qid] = {point.payload["doc_id"]: point.score for point in res.points if point.payload}

            ndcg, _map, recall, precision, mrr, per_query_scores = self._get_evaluation_result(qrels=qrels, results=results)

            # store metrics for _save_to_disk()
            self.save_results_to_instance(results, ndcg, _map, recall, precision, mrr, per_query_scores)

            self._result_printer(ndcg=self.ndcg,
                                mrr=self.mrr,
                                precision=self.precision,
                                _map = self._map,
                                recall=self.recall)

            self._save_to_disk(test_type="hyde")

            return results, ndcg, _map, recall, precision, mrr, per_query_scores

    @override
    def get_run_configurations(self, batch_size: int = 32):
        _, _, qrels = self._load_local_data()
        queries = self._get_hypo_document_dict(self.hypo_queries_path) 
        results = {}
        batch_size = batch_size
        return queries,qrels,results,batch_size
    
    def _get_hypo_document_dict(self, path):
        data = {}
        with open(path, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                data[rec["_id"]] = rec["passage"]
        return data
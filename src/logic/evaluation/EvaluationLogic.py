# this will contain the evaluationlogic from beir
# this classe's methods should be called from testing.py
import json
import logging
import os
from datetime import datetime

from beir import LoggingHandler, util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval

from src.components import EmbeddingModelInstance, QdrantInstance
from src.config.general_config import DATASET_FOLDER_PATHS, RESULT_DIR


class EvaluationLogic:

    def __init__(self, score_function = "cos_sim", vector_store = None, embedding_model_instance = None, target_dataset = "scifact"):
    
        self.score_function = score_function
        self.dateformat = '%Y-%m-%d %H:%M:%S'

        self.vector_store = vector_store or QdrantInstance()
        self.embedding_model = embedding_model_instance or EmbeddingModelInstance()

        self.target_dataset_string = target_dataset.lower()
        self.dataset_path = str(self._get_dataset_path())

        self.results: dict = {}
        self.ndcg: dict[int, float] = {}
        self._map: dict[int, float] = {}
        self.recall: dict[int, float] = {}
        self.precision: dict[int, float] = {}
        self.mrr: dict[int, float] = {}
        self.per_query_scores: dict = {}  # Add this for per-query scores

        # evaluation needs max in k value range to eval correctly across all ranges
        self.k_value_range = [1, 3, 5, 10, 100]
        self.top_k_max = max(self.k_value_range)
        
        #configure logging globally 
        logging.basicConfig(format='%(asctime)s - %(message)s',
                datefmt= self.dateformat,
                level=logging.INFO,
                handlers=[LoggingHandler()]) 
        
    
    def run_evaluation(self, target_collection):

        queries, qrels, results, batch_size = self.get_run_configurations() # if batch size is not set = 32
        collection_name = target_collection

        query_ids = list(qrels.keys())
        query_texts = [queries[qid] for qid in query_ids]        

        for i in range(0, len(query_texts), batch_size):
            batch_texts = query_texts[i:i+batch_size] # delar arrayen [i=0:i+batchsize=32] osv
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

        ndcg, _map, recall, precision, mrr, single_scores = self._get_evaluation_result(qrels=qrels, results=results)

        # store metrics to instance 
        self.save_results_to_instance(results, ndcg, _map, recall, precision, mrr, single_scores)
        # print results
        self._result_printer(ndcg=self.ndcg, 
                             mrr=self.mrr, 
                             precision=self.precision, 
                             _map = self._map, 
                             recall=self.recall)
        # save to disk
        self._save_to_disk()

        return results, ndcg, _map, recall, precision, mrr, single_scores

    def get_run_configurations(self, batch_size: int = 32):
        corpus, queries, qrels = self._load_local_data()
        results = {}
        batch_size = batch_size
        return queries,qrels,results,batch_size

    def save_results_to_instance(self, results, ndcg, _map, recall, precision, mrr, per_query_scores):
        self.results = results
        self.ndcg = ndcg
        self._map = _map
        self.recall = recall
        self.precision = precision
        self.mrr = mrr
        self.per_query_scores = per_query_scores

    # helpers
    def _load_local_data(self):
        """
        Load data from disks 
        """
        corpus, queries, qrels = GenericDataLoader(data_folder=self.dataset_path).load(split="test")
        return corpus, queries, qrels

    def _get_dataset_path(self):
        """
        Return the path of the dataset based on target_dataset.
        Raises an error if dataset not found.
        """
        dataset_path = DATASET_FOLDER_PATHS.get(self.target_dataset_string)
        if not dataset_path:
            raise ValueError(f"Dataset '{self.target_dataset_string}' not found in DATASET_FOLDER_PATH.")
        return dataset_path
    
    def _get_evaluation_result(self, results, qrels):
        """
        Compute BEIR retrieval metrics from results and qrels.
        Args:
            results: {qid: {doc_id: score}}
            qrels: {qid: {doc_id: relevance}}
        Returns:
            ndcg, _map, recall, precision, mrr
        """
        evaluator = EvaluateRetrieval()
        ndcg, _map, recall, precision, single_scores = evaluator.evaluate(
        qrels,
        results,
        k_values= self.k_value_range
        )
        mrr = evaluator.evaluate_custom(
            qrels,
            results,
            k_values=[1, 3, 5, 10],
            metric="mrr"
        )
        return ndcg, _map, recall, precision, mrr, single_scores
    
    def _save_to_disk(self, test_type: str = "base"):

        base_path = RESULT_DIR.get(test_type)
        if base_path is None:
            raise ValueError(f"Unknown test_type: {test_type}")
        try:
            # Create a unique folder per run with readable timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d___%H-%M-%S")
            run_folder = os.path.join(base_path, f"{self.target_dataset_string}_{timestamp}_{test_type}")
            os.makedirs(run_folder, exist_ok=True)

            # Save runfile and results inside this folder
            util.save_runfile(os.path.join(run_folder, f"{self.target_dataset_string}.run.trec"), self.results)
            util.save_results(
                os.path.join(run_folder, f"{self.target_dataset_string}.json"),
                self.ndcg,
                self._map,
                self.recall,
                self.precision,
                self.mrr
            )
            # per query scores
            with open(os.path.join(run_folder, f"{self.target_dataset_string}_per_query_scores.json"), 'w') as f:
                json.dump(self.per_query_scores, f, indent=2)
        except IOError as e:
            print(f"Error saving file:{e}")
    
    def _result_printer(self, ndcg, _map, recall, precision, mrr):
        print(f"BEIR Evaluation for dataset '{self.target_dataset_string}':")
        print("NDCG:", ndcg)
        print("MAP:", _map)
        print("Recall:", recall)
        print("Precision:", precision)
        print("MRR:", mrr)

from src.components.db import QdrantInstance
from src.components.embedding import EmbeddingModelInstance
from src.logic.evaluation import (
    EvaluationLogic,
    EvaluationLogic_HybridSearch,
    EvaluationLogic_HyDE,
    EvaluationLogic_HyPE,
)


def baseEvalutation (target_dataset: str, target_collection: str):
      # Initialize evaluation
    vector_store = QdrantInstance()
    embedding_model = EmbeddingModelInstance()

    evaluator = EvaluationLogic(target_dataset=target_dataset, vector_store=vector_store, embedding_model_instance=embedding_model)

    results, ndcg, _map, recall, precision, mrr, single_score = evaluator.run_evaluation(target_collection=target_collection)

    evaluator.vector_store.close_connection()
    
    # Print metrics
    print("NDCG:", ndcg)
    print("MAP:", _map)
    print("Recall:", recall)
    print("Precision:", precision)
    print("MRR:", mrr)

def hydeEvaluation (target_dataset: str, target_collection: str):
    vector_store = QdrantInstance()
    embedding_model = EmbeddingModelInstance()

    evaluator = EvaluationLogic_HyDE(target_dataset=target_dataset, vector_store=vector_store, embedding_model_instance=embedding_model)

    results, ndcg, _map, recall, precision, mrr, _ = evaluator.run_evaluation(target_collection=target_collection)

    evaluator.vector_store.close_connection()
    
    # Print metrics
    print("NDCG:", ndcg)
    print("MAP:", _map)
    print("Recall:", recall)
    print("Precision:", precision)
    print("MRR:", mrr)

def hypeEvaluation(target_dataset: str, target_collection: str):
    vector_store = QdrantInstance()
    embedding_model = EmbeddingModelInstance()

    evaluator = EvaluationLogic_HyPE(
        target_dataset=target_dataset,
        vector_store=vector_store,
        embedding_model_instance=embedding_model,
    )
    results, ndcg, _map, recall, precision, mrr, _ = evaluator.run_evaluation(
        target_collection=target_collection
    )

    evaluator.vector_store.close_connection()
    
    # Print metrics
    print("NDCG:", ndcg)
    print("MAP:", _map)
    print("Recall:", recall)
    print("Precision:", precision)
    print("MRR:", mrr) 

def hybridsearchEvaluation (target_dataset: str, target_collection: str):
    vector_store = QdrantInstance()
    embedding_model = EmbeddingModelInstance()

    evaluator = EvaluationLogic_HybridSearch(target_dataset=target_dataset, vector_store=vector_store, embedding_model_instance=embedding_model)

    results, ndcg, _map, recall, precision, mrr, _ = evaluator.run_evaluation(target_collection=target_collection)

    evaluator.vector_store.close_connection()
    
    # Print metrics
    print("Hybrid Search Evaluation:")
    print("NDCG:", ndcg)
    print("MAP:", _map)
    print("Recall:", recall)
    print("Precision:", precision)
    print("MRR:", mrr)

if __name__ == "__main__":
  #for s in DATASET_STRINGS:
      baseEvalutation("scifact", "scifact")
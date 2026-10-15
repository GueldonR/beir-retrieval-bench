from src.components.db import QdrantInstance
from src.config.general_config import DATASET_STRINGS
from src.config.run_config import COLLECTIONS, DATASET, EMBEDDING_BATCH_SIZE
from src.logic.ingestion import IngestDataLogic, IngestDataLogicHype


def setup_infra_base():
    setup = QdrantInstance()

    setup.setup_collection(collection_name=COLLECTIONS["base"])

    ingestData = IngestDataLogic(
        qdrant_instance=setup,
        target_dataset=DATASET,
        target_collection=COLLECTIONS["base"],
    )

    ingestData.ingest_data(embedding_batch_size=EMBEDDING_BATCH_SIZE)


def setup_infra_all():
    setup = QdrantInstance()

    try:
        for s in DATASET_STRINGS:
            try:
                setup.setup_collection(collection_name=s)

                ingest_data = IngestDataLogic(
                    qdrant_instance=setup, target_dataset=s, target_collection=s
                )

                ingest_data.ingest_data(embedding_batch_size=20)
            except Exception as e:
                print(f"Failed dataset {s}: {e}")

    finally:
        setup.close_connection()


def setup_infra_HyPE():
    setup = QdrantInstance()

    setup.setup_collection(collection_name=COLLECTIONS["hype"])

    ingestData = IngestDataLogicHype(
        qdrant_instance=setup,
        target_dataset=DATASET,
        target_collection=COLLECTIONS["hype"],
    )

    ingestData.ingest_data(embedding_batch_size=EMBEDDING_BATCH_SIZE)


def setup_infra_all_HyPE():
    setup = QdrantInstance()
    hype_string = "_hype"
    try:
        for s in DATASET_STRINGS:
            try:
                setup.setup_collection(collection_name=s + hype_string)

                ingest_data = IngestDataLogicHype(
                    qdrant_instance=setup,
                    target_dataset=s,
                    target_collection=s + hype_string,
                )

                ingest_data.ingest_data(embedding_batch_size=22)
            except Exception as e:
                print(f"Failed dataset {s}: {e}")

    finally:
        setup.close_connection()


def setup_infra_hybrid():
    setup = QdrantInstance()

    setup.setup_collection(collection_name=COLLECTIONS["hybrid"], hybrid_search=True)

    ingestData = IngestDataLogic(
        qdrant_instance=setup,
        target_dataset=DATASET,
        target_collection=COLLECTIONS["hybrid"],
        hybrid_search=True,
    )

    ingestData.ingest_data()


def setup_infra_all_hybrid():
    setup = QdrantInstance()
    hybrid_string = "_hybrid"

    try:
        for s in DATASET_STRINGS:
            try:
                collection_name = s + hybrid_string
                setup.setup_collection(
                    collection_name=collection_name, hybrid_search=True
                )

                ingest_data = IngestDataLogic(
                    qdrant_instance=setup,
                    target_dataset=s,
                    target_collection=collection_name,
                    hybrid_search=True,
                )

                ingest_data.ingest_data()
            except Exception as e:
                print(f"Failed dataset {s}: {e}")

    finally:
        setup.close_connection()


if __name__ == "__main__":
    setup_infra_base()
    setup_infra_hybrid()

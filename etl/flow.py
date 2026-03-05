# flows/orchestrator.py
import os
from prefect import flow
from silver_etl import silver_etl_task
from gold_etl import gold_etl_task
from ingestion.opensky_ingest_flow import opensky_ingest_flow
from training.train_model import train_models_flow
from predict.predict import predict_positions_flow

@flow(name="flight_pipeline_flow")
def flight_pipeline():
    GCS_BUCKET = os.getenv("GCS_BUCKET")
    GCP_PROJECT = os.getenv("GCP_PROJECT_ID")
    BQ_DATASET = os.getenv("BQ_DATASET_ID")
    BQ_PRED_TABLE = os.getenv("BQ_PRED_TABLE", "gold_predictions")

    # Step 1: Bronze
    opensky_ingest_flow()

    # Step 2: Silver
    silver_etl_task(GCS_BUCKET)

    # Step 3: Train models
    # train_models_flow()

    # Step 4: Predict plane position
    if os.getenv("CI", "false").lower() != "true":
        predict_positions_flow()

    # Step : Gold
    gold_etl_task(GCS_BUCKET, GCP_PROJECT, BQ_DATASET, BQ_PRED_TABLE)

if __name__ == "__main__":
    flight_pipeline()
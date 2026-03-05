import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, current_timestamp, expr, to_date
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import RandomForestRegressionModel

# -----------------------------
# Env vars
# -----------------------------
GCS_BUCKET = os.getenv("GCS_BUCKET")
GCP_PROJECT = os.getenv("GCP_PROJECT_ID")
BQ_DATASET = os.getenv("BQ_DATASET_ID")
BQ_PRED_TABLE = os.getenv("BQ_PRED_TABLE", "gold_predictions")

if not all([GCS_BUCKET, GCP_PROJECT, BQ_DATASET]):
    raise RuntimeError("Missing required env vars")

spark = SparkSession.builder \
    .appName("BatchPredict") \
    .config("spark.jars.packages", "com.google.cloud.bigdataoss:gcs-connector:hadoop3-2.2.22") \
    .getOrCreate()

# -----------------------------
# Read Silver ETL data
# -----------------------------
silver_path = f"gs://{GCS_BUCKET}/silver/flight_states/"
silver_df = spark.read.parquet(silver_path)

# Filter last hour (adjust interval as needed)
silver_df = silver_df.filter(col("ingest_ts") > current_timestamp() - expr("INTERVAL 1 HOUR"))

# -----------------------------
# Features
# -----------------------------
feature_cols = ["lat", "lon", "baro_altitude", "velocity", "heading",
                "vertical_rate", "accel", "turn_rate"]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
feature_df = assembler.transform(silver_df)

# -----------------------------
# Load pretrained models
# -----------------------------
model_lat = RandomForestRegressionModel.load(f"gs://{GCS_BUCKET}/models/rf_lat_model")
model_lon = RandomForestRegressionModel.load(f"gs://{GCS_BUCKET}/models/rf_lon_model")
model_alt = RandomForestRegressionModel.load(f"gs://{GCS_BUCKET}/models/rf_alt_model")

# -----------------------------
# Predict
# -----------------------------
pred_lat = model_lat.transform(feature_df).withColumnRenamed("prediction", "pred_lat")
pred_lon = model_lon.transform(feature_df).withColumnRenamed("prediction", "pred_lon")
pred_alt = model_alt.transform(feature_df).withColumnRenamed("prediction", "pred_alt")

# Merge predictions
pred_df = pred_lat.select("icao24", "time_position", "ingest_ts", "pred_lat") \
    .join(pred_lon.select("icao24", "time_position", "pred_lon"), on=["icao24", "time_position"]) \
    .join(pred_alt.select("icao24", "time_position", "pred_alt"), on=["icao24", "time_position"])

# Partition column
pred_df = pred_df.withColumn("ingest_date", to_date("ingest_ts"))

# -----------------------------
# Write to GCS
# -----------------------------
gcs_pred_path = f"gs://{GCS_BUCKET}/gold/predictions/"
pred_df.write.mode("append").partitionBy("ingest_date").parquet(gcs_pred_path)

# -----------------------------
# Write to BigQuery
# -----------------------------
pred_df.write.mode("append").format("bigquery") \
    .option("table", f"{GCP_PROJECT}.{BQ_DATASET}.{BQ_PRED_TABLE}") \
    .save()

spark.stop()
print("Predictions completed successfully.")
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, current_timestamp, expr
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import RandomForestRegressionModel
from google.cloud import storage

GCS_BUCKET = os.getenv("GCS_BUCKET")
GCP_PROJECT = os.getenv("GCP_PROJECT_ID")
BQ_DATASET = os.getenv("BQ_DATASET_ID")
BQ_PRED_TABLE = os.getenv("BQ_PRED_TABLE", "gold_predictions")

if not all([GCS_BUCKET, GCP_PROJECT, BQ_DATASET]):
    raise RuntimeError("Missing required env vars")

spark = SparkSession.builder.appName("GoldPredictETL_Incremental").getOrCreate()

# -----------------------------
# Step 1: Track processed Silver partitions
# -----------------------------
client = storage.Client()
bucket = client.bucket(GCS_BUCKET)
processed_blob = bucket.blob("gold/_processed_partitions.txt")

try:
    processed_list = processed_blob.download_as_text().splitlines()
except Exception:
    processed_list = []

# -----------------------------
# Step 2: List Silver partitions
# -----------------------------
blobs = list(bucket.list_blobs(prefix="silver/flight_states/"))
partitions = sorted(set([b.name.rsplit("/", 4)[0] for b in blobs]))  # get partition folder paths
new_partitions = [p for p in partitions if p not in processed_list]

if not new_partitions:
    print("No new Silver partitions to process for prediction.")
    spark.stop()
    exit(0)

# -----------------------------
# Step 3: Read new Silver data
# -----------------------------
silver_paths = [f"gs://{GCS_BUCKET}/{p}/*" for p in new_partitions]
silver_df = spark.read.parquet(*silver_paths)

# -----------------------------
# Step 4: Feature engineering
# -----------------------------
w = Window.partitionBy("icao24").orderBy("time_position")
feature_df = silver_df.withColumn("prev_lat", lag("lat", 1).over(w)) \
                      .withColumn("prev_lon", lag("lon", 1).over(w)) \
                      .withColumn("prev_velocity", lag("velocity", 1).over(w)) \
                      .withColumn("prev_heading", lag("heading", 1).over(w)) \
                      .withColumn("time_diff_sec", (unix_timestamp("time_position") - unix_timestamp(lag("time_position", 1).over(w)))) \
                      .withColumn("accel", (col("velocity") - col("prev_velocity")) / col("time_diff_sec")) \
                      .withColumn("turn_rate", (col("heading") - col("prev_heading")) / col("time_diff_sec")) \
                      .filter(col("prev_lat").isNotNull())

feature_cols = ["lat", "lon", "baro_altitude", "velocity", "heading", "vertical_rate", "accel", "turn_rate"]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
feature_df = assembler.transform(feature_df)

# -----------------------------
# Step 5: Load models and predict
# -----------------------------
models = {
    "pred_lat": f"gs://{GCS_BUCKET}/models/rf_lat_model",
    "pred_lon": f"gs://{GCS_BUCKET}/models/rf_lon_model",
    "pred_alt": f"gs://{GCS_BUCKET}/models/rf_alt_model"
}

predictions = {}
for col_name, path in models.items():
    model = RandomForestRegressionModel.load(path)
    pred_df = model.transform(feature_df).withColumnRenamed("prediction", col_name)
    predictions[col_name] = pred_df.select("icao24", "time_position", "ingest_ts", col_name)

# Join predictions
pred_df = predictions["pred_lat"].join(predictions["pred_lon"], on=["icao24", "time_position"]) \
                                 .join(predictions["pred_alt"], on=["icao24", "time_position"])
pred_df = pred_df.withColumn("ingest_date", col("ingest_ts").cast("date"))

# -----------------------------
# Step 6: Write to GCS and BigQuery
# -----------------------------
# GCS
gcs_pred_path = f"gs://{GCS_BUCKET}/gold/predictions/"
pred_df.write.mode("append").partitionBy("ingest_date").parquet(gcs_pred_path)

# BigQuery
pred_df.write.mode("append").format("bigquery") \
    .option("table", f"{GCP_PROJECT}.{BQ_DATASET}.{BQ_PRED_TABLE}") \
    .save()

# -----------------------------
# Step 7: Update processed partitions
# -----------------------------
processed_list.extend(new_partitions)
processed_blob.upload_from_string("\n".join(processed_list))

spark.stop()
print(f"Processed {len(new_partitions)} new Silver partitions for prediction.")
import os
from pyspark.sql import SparkSession
from pyspark.sql.window import Window
from pyspark.sql.functions import col, lag, unix_timestamp, current_timestamp, expr, year
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import RandomForestRegressionModel
from prefect import task, get_run_logger

@task(retries=3, retry_delay_seconds=10)
def gold_etl_task(gcs_bucket: str, gcp_project: str, bq_dataset: str, bq_table: str = "gold_predictions"):
    logger = get_run_logger()
    logger.info("Starting Gold ETL + Predictions...")

    # Full load flag
    full_load = os.getenv("GOLD_FULL_LOAD", "false").lower() == "true"
    if full_load:
        logger.info("Running GOLD full historical load.")
    else:
        logger.info("Running GOLD incremental load (last 1 hour).")

    spark = SparkSession.builder \
        .appName("GoldPredictETL") \
        .config("spark.jars", "jars/gcs-connector-hadoop3-2.2.11-shaded.jar") \
        .config("spark.hadoop.fs.gs.impl", "com.google.cloud.hadoop.fs.gcs.GoogleHadoopFileSystem") \
        .config("spark.hadoop.fs.AbstractFileSystem.gs.impl", "com.google.cloud.hadoop.fs.gcs.GoogleHadoopFS") \
        .getOrCreate()

    silver_path = f"gs://{gcs_bucket}/silver/flight_states/"
    silver_df = spark.read.parquet(silver_path)

    # Incremental filter
    if not full_load:
        silver_df = silver_df.withColumn("ingest_ts", col("ingest_ts").cast("timestamp"))
        silver_df = silver_df.filter(col("ingest_ts") > current_timestamp() - expr("INTERVAL 1 HOUR"))

    # Feature engineering
    w = Window.partitionBy("icao24").orderBy("time_position")
    feature_df = silver_df \
        .withColumn("prev_lat", lag("lat", 1).over(w)) \
        .withColumn("prev_lon", lag("lon", 1).over(w)) \
        .withColumn("prev_velocity", lag("velocity", 1).over(w)) \
        .withColumn("prev_heading", lag("heading", 1).over(w)) \
        .withColumn("time_diff_sec",
                    (unix_timestamp("time_position") - unix_timestamp(lag("time_position", 1).over(w)))) \
        .withColumn("accel", (col("velocity") - col("prev_velocity")) / col("time_diff_sec")) \
        .withColumn("turn_rate", (col("heading") - col("prev_heading")) / col("time_diff_sec"))

    feature_df = feature_df.filter(col("prev_lat").isNotNull())

    feature_cols = ["lat", "lon", "baro_altitude", "velocity", "heading",
                    "vertical_rate", "accel", "turn_rate"]
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    feature_df = assembler.transform(feature_df)

    # Load models
    model_lat = RandomForestRegressionModel.load(f"gs://{gcs_bucket}/models/rf_lat_model")
    model_lon = RandomForestRegressionModel.load(f"gs://{gcs_bucket}/models/rf_lon_model")
    model_alt = RandomForestRegressionModel.load(f"gs://{gcs_bucket}/models/rf_alt_model")

    pred_lat = model_lat.transform(feature_df).withColumnRenamed("prediction", "pred_lat")
    pred_lon = model_lon.transform(feature_df).withColumnRenamed("prediction", "pred_lon")
    pred_alt = model_alt.transform(feature_df).withColumnRenamed("prediction", "pred_alt")

    pred_df = pred_lat.select("icao24", "time_position", "ingest_ts", "pred_lat") \
        .join(pred_lon.select("icao24", "time_position", "pred_lon"), on=["icao24", "time_position"]) \
        .join(pred_alt.select("icao24", "time_position", "pred_alt"), on=["icao24", "time_position"])

    pred_df = pred_df.withColumn("ingest_date", col("ingest_ts").cast("date"))

    # Write to BigQuery only if full_load or incremental mode explicitly wants it
    write_mode = "overwrite" if full_load else "append"
    pred_df.write.mode(write_mode).format("bigquery") \
        .option("table", f"{gcp_project}.{bq_dataset}.{bq_table}").save()
    logger.info(f"Written to BigQuery table {bq_dataset}.{bq_table} (mode={write_mode})")

    # Write to GCS gold
    gcs_pred_path = f"gs://{gcs_bucket}/gold/predictions/"
    pred_df.write.mode(write_mode).partitionBy("ingest_date").parquet(gcs_pred_path)
    logger.info(f"Written predictions to GCS {gcs_pred_path} (mode={write_mode})")

    logger.info("Gold ETL + Predictions complete.")
    spark.stop()
    return gcs_pred_path
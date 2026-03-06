import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, current_timestamp, expr
from prefect import task, get_run_logger

@task(retries=3, retry_delay_seconds=10)
def gold_etl_task(gcs_bucket: str, gcp_project: str, bq_dataset: str, bq_table: str = "gold_predictions"):
    logger = get_run_logger()
    logger.info("Starting GOLD ETL (CI-safe, no ML predictions)...")

    full_load = os.getenv("SILVER_FULL_LOAD", "false").lower() == "true"
    logger.info("Running GOLD full load." if full_load else "Running GOLD incremental load (last 1 hour).")

    bq_jar_path = os.path.abspath("jars/spark-bigquery-with-dependencies_2.12-0.30.0.jar")
    if not os.path.exists(bq_jar_path):
        raise FileNotFoundError(f"BigQuery jar not found at {bq_jar_path}")
    gcs_jar_path = os.path.abspath("jars/gcs-connector-hadoop3-2.2.11-shaded.jar")

    spark = SparkSession.builder \
        .appName("GoldETL_CI_Safe") \
        .config("spark.jars", f"{gcs_jar_path},{bq_jar_path}") \
        .config("spark.jars.packages", "com.google.cloud.spark:spark-bigquery-with-dependencies_2.12:0.30.0") \
        .config("spark.hadoop.fs.gs.impl", "com.google.cloud.hadoop.fs.gcs.GoogleHadoopFileSystem") \
        .config("spark.hadoop.fs.AbstractFileSystem.gs.impl", "com.google.cloud.hadoop.fs.gcs.GoogleHadoopFS") \
        .getOrCreate()
    silver_path = f"gs://{gcs_bucket}/silver/flight_states/"
    try:
        silver_df = spark.read.option("basePath", silver_path) \
            .option("mergeSchema", True) \
            .option("recursiveFileLookup", True) \
            .parquet(silver_path)
        logger.info(f"Loaded Silver data from {silver_path}")
    except Exception as e:
        logger.error(f"Failed to read Silver data: {e}")
        spark.stop()
        return None

    if not full_load:
        if "ingest_ts" in silver_df.columns:
            silver_df = silver_df.withColumn("ingest_ts", col("ingest_ts").cast("timestamp"))
            silver_df = silver_df.filter(col("ingest_ts") > current_timestamp() - expr("INTERVAL 1 HOUR"))
            logger.info(f"Filtered Silver data for incremental load. Rows remaining: {silver_df.count()}")
        else:
            logger.warning("No 'ingest_ts' column found for incremental filtering.")

    # Avoid writing empty DataFrame
    row_count = silver_df.count()
    if row_count == 0:
        logger.warning("Silver DataFrame is empty. Skipping GOLD write to GCS and BigQuery.")
        spark.stop()
        return None

    silver_df = silver_df.withColumn("ingest_date", col("ingest_ts").cast("date"))

    write_mode = "overwrite" if full_load else "append"
    gcs_gold_path = f"gs://{gcs_bucket}/gold/flight_states/"
    silver_df.write.mode(write_mode).partitionBy("ingest_date").parquet(gcs_gold_path)
    logger.info(f"Written GOLD data to GCS {gcs_gold_path} (mode={write_mode})")

    try:
        silver_df.write \
            .format("com.google.cloud.spark.bigquery") \
            .option("table", f"{gcp_project}.{bq_dataset}.{bq_table}") \
            .option("temporaryGcsBucket", gcs_bucket) \
            .mode(write_mode) \
            .save()
    except Exception as e:
        import traceback
        logger.error("BigQuery write failed:\n" + traceback.format_exc())
    logger.info(f"Written GOLD data to BigQuery table {bq_dataset}.{bq_table} (mode={write_mode})")

    spark.stop()
    logger.info("GOLD ETL (CI-safe) complete.")
    return gcs_gold_path
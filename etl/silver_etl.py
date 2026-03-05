import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_timestamp, year, month, dayofmonth, hour, input_file_name
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, BooleanType
from prefect import task, get_run_logger

@task(retries=3, retry_delay_seconds=10)
def silver_etl_task(gcs_bucket: str):
    logger = get_run_logger()
    logger.info("Starting Silver ETL...")

    full_load = os.getenv("SILVER_FULL_LOAD", "false").lower() == "true"
    if full_load:
        logger.info("Running SILVER full historical load.")
    else:
        logger.info("Running SILVER incremental load (last 1 hour).")

    schema = StructType([
        StructField("ingest_ts", StringType(), True),
        StructField("icao24", StringType(), True),
        StructField("callsign", StringType(), True),
        StructField("origin_country", StringType(), True),
        StructField("time_position", StringType(), True),
        StructField("last_contact", StringType(), True),
        StructField("lon", DoubleType(), True),
        StructField("lat", DoubleType(), True),
        StructField("baro_altitude", DoubleType(), True),
        StructField("on_ground", BooleanType(), True),
        StructField("velocity", DoubleType(), True),
        StructField("heading", DoubleType(), True),
        StructField("vertical_rate", DoubleType(), True),
    ])

    spark = SparkSession.builder \
        .appName("SilverETL") \
        .config("spark.jars", "jars/gcs-connector-hadoop3-2.2.11-shaded.jar") \
        .config("spark.hadoop.fs.gs.impl", "com.google.cloud.hadoop.fs.gcs.GoogleHadoopFileSystem") \
        .config("spark.hadoop.fs.AbstractFileSystem.gs.impl", "com.google.cloud.hadoop.fs.gcs.GoogleHadoopFS") \
        .getOrCreate()

    bronze_path = f"gs://{gcs_bucket}/bronze/"
    raw_df = spark.read.schema(schema).json(bronze_path)

    if not full_load:
        from pyspark.sql.functions import current_timestamp, expr
        raw_df = raw_df.withColumn("ingest_ts", to_timestamp("ingest_ts"))
        raw_df = raw_df.filter(col("ingest_ts") > current_timestamp() - expr("INTERVAL 1 HOUR"))

    raw_df = raw_df.withColumn("source_file", input_file_name())

    silver_df = raw_df \
        .withColumn("time_position", to_timestamp("time_position")) \
        .withColumn("last_contact", to_timestamp("last_contact")) \
        .filter(col("icao24").isNotNull()) \
        .filter(col("lat").between(-90, 90) & col("lon").between(-180, 180)) \
        .dropDuplicates(["icao24", "time_position"])

    silver_df = silver_df \
        .withColumn("year", year("ingest_ts")) \
        .withColumn("month", month("ingest_ts")) \
        .withColumn("day", dayofmonth("ingest_ts")) \
        .withColumn("hour", hour("ingest_ts"))

    silver_output = f"gs://{gcs_bucket}/silver/flight_states/"
    write_mode = "overwrite" if full_load else "append"
    silver_df.write.mode(write_mode).partitionBy("year", "month", "day", "hour").parquet(silver_output)

    logger.info(f"Silver ETL complete. Written to {silver_output} (mode={write_mode}).")
    spark.stop()
    return silver_output
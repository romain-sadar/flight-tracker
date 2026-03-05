import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_timestamp, year, month, dayofmonth, hour, input_file_name
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, BooleanType
from google.cloud import storage

GCS_BUCKET = os.getenv("GCS_BUCKET")
if not GCS_BUCKET:
    raise RuntimeError("Set GCS_BUCKET environment variable")

# Initialize Spark
spark = SparkSession.builder \
    .appName("SilverETL_Incremental") \
    .config("spark.jars", "etl/jars/gcs-connector-hadoop3-2.2.22-shaded.jar") \
    .config("spark.hadoop.fs.gs.impl", "com.google.cloud.hadoop.fs.gcs.GoogleHadoopFileSystem") \
    .config("spark.hadoop.fs.AbstractFileSystem.gs.impl", "com.google.cloud.hadoop.fs.gcs.GoogleHadoopFS") \
    .getOrCreate()

# Schema definition
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

# List Bronze files
client = storage.Client()
bucket = client.bucket(GCS_BUCKET)
blobs = list(bucket.list_blobs(prefix="bronze/"))
all_files = sorted([b.name for b in blobs if b.name.endswith(".ndjson")])

# Read list of already processed files
processed_file = f"gs://{GCS_BUCKET}/silver/_processed_files.txt"
try:
    processed_blobs = bucket.blob("_processed_files.txt")
    processed_list = processed_blobs.download_as_text().splitlines()
except Exception:
    processed_list = []

# Keep only new files
new_files = [f"gs://{GCS_BUCKET}/{f}" for f in all_files if f not in processed_list]
if not new_files:
    print("No new Bronze files to process.")
    spark.stop()
    exit(0)

# Read new Bronze files
raw_df = spark.read.schema(schema).json(new_files)
raw_df = raw_df.withColumn("source_file", input_file_name())

# Transform
silver_df = raw_df \
    .withColumn("ingest_ts", to_timestamp("ingest_ts")) \
    .withColumn("time_position", to_timestamp("time_position")) \
    .withColumn("last_contact", to_timestamp("last_contact")) \
    .filter(col("icao24").isNotNull()) \
    .filter(col("lat").between(-90, 90) & col("lon").between(-180, 180)) \
    .dropDuplicates(["icao24", "time_position"]) \
    .withColumn("year", year("ingest_ts")) \
    .withColumn("month", month("ingest_ts")) \
    .withColumn("day", dayofmonth("ingest_ts")) \
    .withColumn("hour", hour("ingest_ts"))

# Write Silver Parquet partitioned
silver_output = f"gs://{GCS_BUCKET}/silver/flight_states/"
silver_df.write.mode("append").partitionBy("year", "month", "day", "hour").parquet(silver_output)

# Update processed files
processed_blobs.upload_from_string("\n".join(all_files))
print(f"Processed {len(new_files)} new Bronze files.")

spark.stop()
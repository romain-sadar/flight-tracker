import os
from pyspark.sql import SparkSession
from pyspark.sql.window import Window
from pyspark.sql.functions import col, lag, unix_timestamp, when
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import RandomForestRegressionModel
from prefect import flow, task, get_run_logger


@task(retries=3, retry_delay_seconds=10)
def predict_positions(gcs_bucket: str):

    logger = get_run_logger()
    logger.info("Starting trajectory prediction...")

    spark = SparkSession.builder \
        .appName("PredictAircraftPositions") \
        .config("spark.jars.packages", "com.google.cloud.bigdataoss:gcs-connector:hadoop3-2.2.22") \
        .getOrCreate()

    silver_path = f"gs://{gcs_bucket}/silver/flight_states/"
    df = spark.read.parquet(silver_path)

    logger.info(f"Loaded {df.count()} rows from silver.")

    # window for aircraft history
    w = Window.partitionBy("icao24").orderBy("time_position")

    df = df.withColumn("prev_velocity", lag("velocity", 1).over(w)) \
        .withColumn("prev_heading", lag("heading", 1).over(w)) \
        .withColumn("prev_time", lag("time_position", 1).over(w)) \
        .withColumn("time_diff_sec",
                    unix_timestamp("time_position") - unix_timestamp("prev_time")) \
        .withColumn(
            "accel",
            when(col("time_diff_sec") != 0,
                 (col("velocity") - col("prev_velocity")) / col("time_diff_sec")
                 ).otherwise(0.0)
        ) \
        .withColumn(
            "turn_rate",
            when(col("time_diff_sec") != 0,
                 (col("heading") - col("prev_heading")) / col("time_diff_sec")
                 ).otherwise(0.0)
        )

    feature_cols = [
        "lat",
        "lon",
        "baro_altitude",
        "velocity",
        "heading",
        "vertical_rate",
        "accel",
        "turn_rate"
    ]

    df = df.dropna(subset=feature_cols)

    assembler = VectorAssembler(
        inputCols=feature_cols,
        outputCol="features",
        handleInvalid="skip"
    )

    data = assembler.transform(df)

    # Load trained models
    lat_model = RandomForestRegressionModel.load(
        f"gs://{gcs_bucket}/models/rf_lat_model"
    )

    lon_model = RandomForestRegressionModel.load(
        f"gs://{gcs_bucket}/models/rf_lon_model"
    )

    alt_model = RandomForestRegressionModel.load(
        f"gs://{gcs_bucket}/models/rf_alt_model"
    )

    logger.info("Models loaded.")

    # predictions
    pred_lat = lat_model.transform(data).withColumnRenamed(
        "prediction", "predicted_lat")

    pred_lon = lon_model.transform(pred_lat).withColumnRenamed(
        "prediction", "predicted_lon")

    pred_alt = alt_model.transform(pred_lon).withColumnRenamed(
        "prediction", "predicted_alt")

    result = pred_alt.select(
        "icao24",
        "time_position",
        "lat",
        "lon",
        "baro_altitude",
        "predicted_lat",
        "predicted_lon",
        "predicted_alt"
    )

    output_path = f"gs://{gcs_bucket}/gold/trajectory_predictions/"

    result.write.mode("overwrite").parquet(output_path)

    logger.info(f"Predictions saved to {output_path}")

    spark.stop()

    return output_path


@flow(name="predict_positions_flow")
def predict_positions_flow():

    gcs_bucket = os.getenv("GCS_BUCKET")

    if not gcs_bucket:
        raise RuntimeError("Missing GCS_BUCKET environment variable")

    predict_positions(gcs_bucket)


if __name__ == "__main__":
    predict_positions_flow()
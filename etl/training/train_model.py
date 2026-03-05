import os
from pyspark.sql import SparkSession
from pyspark.sql.window import Window
from pyspark.sql.functions import col, lead, lag, unix_timestamp, when
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from prefect import flow, task, get_run_logger

PREDICTION_HORIZON_ROWS = 5


@task(retries=3, retry_delay_seconds=10)
def train_rf_models(gcs_bucket: str):
    logger = get_run_logger()
    logger.info("Starting training of RandomForest models...")

    spark = SparkSession.builder \
        .appName("TrainModels") \
        .config("spark.jars.packages", "com.google.cloud.bigdataoss:gcs-connector:hadoop3-2.2.22") \
        .getOrCreate()

    silver_path = f"gs://{gcs_bucket}/silver/flight_states/"
    df = spark.read.parquet(silver_path)
    logger.info(f"Read {df.count()} rows from silver ETL.")

    # Compute lag features for accel and turn_rate
    w = Window.partitionBy("icao24").orderBy("time_position")

    df = df \
        .withColumn("prev_lat", lag("lat", 1).over(w)) \
        .withColumn("prev_lon", lag("lon", 1).over(w)) \
        .withColumn("prev_alt", lag("baro_altitude", 1).over(w)) \
        .withColumn("prev_velocity", lag("velocity", 1).over(w)) \
        .withColumn("prev_heading", lag("heading", 1).over(w)) \
        .withColumn("prev_time", lag("time_position", 1).over(w)) \
        .withColumn("time_diff_sec", unix_timestamp("time_position") - unix_timestamp("prev_time")) \
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
        ) \
        .withColumn(
            "altitude_change",
            col("baro_altitude") - col("prev_alt")
        )

    # Compute target columns: next lat/lon/alt
    logger.info(f"Predicting aircraft position {PREDICTION_HORIZON_ROWS} observations ahead (~30 seconds).")
    df = df.withColumn("future_lat", lead("lat", PREDICTION_HORIZON_ROWS).over(w)) \
        .withColumn("future_lon", lead("lon", PREDICTION_HORIZON_ROWS).over(w)) \
        .withColumn("future_alt", lead("baro_altitude", PREDICTION_HORIZON_ROWS).over(w))

    # Keep only rows where features and targets exist
    df = df.filter(
        col("accel").isNotNull() &
        col("turn_rate").isNotNull() &
        col("future_lat").isNotNull() &
        col("future_lon").isNotNull() &
        col("future_alt").isNotNull()
    )
    df = df.cache()
    logger.info(f"{df.count()} rows available for training after filtering.")

    feature_cols = [
        "lat",
        "lon",
        "baro_altitude",

        "prev_lat",
        "prev_lon",
        "prev_alt",

        "velocity",
        "prev_velocity",

        "heading",
        "prev_heading",

        "vertical_rate",

        "accel",
        "turn_rate",
        "altitude_change",
    ]
    
    df = df.dropna(subset=feature_cols + [
        "future_lat",
        "future_lon",
        "future_alt"
    ])

    assembler = VectorAssembler(
        inputCols=feature_cols,
        outputCol="features",
        handleInvalid="skip"
    )
    data = assembler.transform(df)

    train, test = data.randomSplit([0.8, 0.2], seed=42)
    logger.info(f"Split into {train.count()} train rows and {test.count()} test rows.")

    targets = [("future_lat", "lat"), ("future_lon", "lon"), ("future_alt", "alt")]
    for label_col, name in targets:
        logger.info(f"Training model for {label_col}...")
        rf = RandomForestRegressor(featuresCol="features", labelCol=label_col, numTrees=50)
        model = rf.fit(train)

        predictions = model.transform(test)
        evaluator = RegressionEvaluator(labelCol=label_col, predictionCol="prediction", metricName="rmse")
        rmse = evaluator.evaluate(predictions)
        logger.info(f"RMSE for {name}: {rmse}")

        model_path = f"gs://{gcs_bucket}/models/rf_{name}_model"
        model.write().overwrite().save(model_path)
        logger.info(f"Saved model to {model_path}")

    spark.stop()
    logger.info("Training complete for all models.")
    return True

@flow(name="train_models_flow")
def train_models_flow():
    gcs_bucket = os.getenv("GCS_BUCKET")
    if not gcs_bucket:
        raise RuntimeError("Missing GCS_BUCKET environment variable")
    train_rf_models(gcs_bucket)

if __name__ == "__main__":
    train_models_flow()
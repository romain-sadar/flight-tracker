import os
from pyspark.sql import SparkSession
from pyspark.sql.window import Window
from pyspark.sql.functions import col, lead
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator

# -----------------------------
# Env vars
# -----------------------------
GCS_BUCKET = os.getenv("GCS_BUCKET")
if not GCS_BUCKET:
    raise RuntimeError("Set GCS_BUCKET environment variable")

spark = SparkSession.builder \
    .appName("TrainModel") \
    .config("spark.jars.packages", "com.google.cloud.bigdataoss:gcs-connector:hadoop3-2.2.22") \
    .getOrCreate()

# -----------------------------
# Read Silver ETL (with lag features)
# -----------------------------
silver_path = f"gs://{GCS_BUCKET}/silver/flight_states/"
df = spark.read.parquet(silver_path)

# -----------------------------
# Compute targets: next position
# -----------------------------
w = Window.partitionBy("icao24").orderBy("time_position")
df = df.withColumn("future_lat", lead("lat", 1).over(w)) \
       .withColumn("future_lon", lead("lon", 1).over(w)) \
       .withColumn("future_alt", lead("baro_altitude", 1).over(w))

# Keep rows where target exists
df = df.filter(col("future_lat").isNotNull())

# -----------------------------
# Feature columns
# -----------------------------
feature_cols = ["lat", "lon", "baro_altitude", "velocity", "heading",
                "vertical_rate", "accel", "turn_rate"]

assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
data = assembler.transform(df)

# -----------------------------
# Split train/test
# -----------------------------
train, test = data.randomSplit([0.8, 0.2], seed=42)

# -----------------------------
# Train a RandomForest for each target
# -----------------------------
targets = [("future_lat", "lat"), ("future_lon", "lon"), ("future_alt", "alt")]
for label_col, name in targets:
    print(f"Training model for {label_col}...")
    rf = RandomForestRegressor(featuresCol="features", labelCol=label_col, numTrees=50)
    model = rf.fit(train)

    # Evaluate
    predictions = model.transform(test)
    evaluator = RegressionEvaluator(labelCol=label_col, predictionCol="prediction", metricName="rmse")
    rmse = evaluator.evaluate(predictions)
    print(f"RMSE for {name}: {rmse}")

    # Save model to GCS
    model_path = f"gs://{GCS_BUCKET}/models/rf_{name}_model"
    model.write().overwrite().save(model_path)
    print(f"Saved model to {model_path}")

spark.stop()
print("Training complete for all models.")
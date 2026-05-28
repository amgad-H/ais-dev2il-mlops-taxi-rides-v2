import sys
import logging
import mlflow
import json
import pickle
import os
from model_trainings import (
  train_random_forest_classifier,
  train_random_forest_classifier_v2,
  train_logistic_regression_classifier,
)

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATA_FILE = "example-data/data/taxi-rides-training-data.parquet"

REQUIRED_ENV_VARS = ["MLFLOW_TRACKING_URI", "MLFLOW_TRACKING_USERNAME", "MLFLOW_TRACKING_PASSWORD"]

def check_env_vars() -> None:
    missing = [var for var in REQUIRED_ENV_VARS if not os.environ.get(var)]
    if missing:
        logger.error(f"Missing required environment variables: {', '.join(missing)}")
        sys.exit(1)

def train_model(model_type: str):
  check_env_vars()
  logger.info("Training outlier detection classifier")

  valid_model_types = ['random_forest', 'random_forest_v2', 'logistic_regression']
  if model_type not in valid_model_types:
    raise ValueError(f"Unknown model_type '{model_type}'. Valid options: {', '.join(valid_model_types)}")

  metadata_output_file = f"models/{model_type}.metadata.json"

  mlflow.set_experiment(model_type)
  mlflow.autolog()

  with mlflow.start_run():
    if model_type == 'random_forest':
      model, metadata = train_random_forest_classifier(DATA_FILE)
    elif model_type == 'random_forest_v2':
      model, metadata = train_random_forest_classifier_v2(DATA_FILE)
    elif model_type == 'logistic_regression':
      model, metadata = train_logistic_regression_classifier(DATA_FILE)
    logger.info("Model training completed")
    # log some custom metrics
    for false_key, false_value in metadata["False"].items():
      mlflow.log_metric(f"False_{false_key}", false_value)
    for false_key, false_value in metadata["True"].items():
      mlflow.log_metric(f"True_{false_key}", false_value)
    mlflow.log_artifact(metadata_output_file)
  os.makedirs("models", exist_ok=True)

  logger.info(f"Writing metadata to: {metadata_output_file}")
  with open(metadata_output_file, "w") as metadata_file:
    json.dump(metadata, metadata_file, indent=4)

  model_output_file = f"models/{model_type}.pkl"
  logger.info(f"Storing model to: {model_output_file}")
  with open(model_output_file, "wb") as f:
    pickle.dump(model, f)

  logger.info(metadata)

  import pandas as pd

  with open("models/random_forest.pkl", "rb") as f:
    model = pickle.load(f)

  rides = pd.DataFrame([
    {"ride_time": 2, "trip_distance": 100},  # 2-second, 100 km trip
    {"ride_time": 1800, "trip_distance": 5},  # 30-minute, 5 km trip
    {"ride_time": 0, "trip_distance": 50},  # Instant 50 km trip
  ])

  predictions = model.predict(rides)
  print(predictions)

if __name__ == "__main__":
  model_type = sys.argv[1]  # random_forest, random_forest_v2, logistic_regression
  train_model(model_type)


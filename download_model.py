"""
Downloads a model from the MLFlow model registry and stores it as outlier_detection_model.pkl.

Usage:
    uv run download_model.py

Required environment variables:
    MLFLOW_TRACKING_URI      - MLFlow tracking server URL
    MLFLOW_TRACKING_USERNAME - MLFlow username
    MLFLOW_TRACKING_PASSWORD - MLFlow password

The model version is read from .model-version.
"""
import pickle
import logging
import os
import sys
import mlflow

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

REQUIRED_ENV_VARS = ["MLFLOW_TRACKING_URI", "MLFLOW_TRACKING_USERNAME", "MLFLOW_TRACKING_PASSWORD"]

MODEL_VERSION_FILE = ".model-version"
MODEL_NAME = "outlier-detection"
OUTPUT_FILE = "outlier_detection_model.pkl"


def check_env_vars() -> None:
    missing = [var for var in REQUIRED_ENV_VARS if not os.environ.get(var)]
    if missing:
        logger.error(f"Missing required environment variables: {', '.join(missing)}")
        sys.exit(1)


def main() -> None:
    check_env_vars()
    version = open(MODEL_VERSION_FILE).read().strip()
    model_uri = f"models:/{MODEL_NAME}@champion"
    logger.info(f"Downloading model from: {model_uri}")

    model = mlflow.sklearn.load_model(model_uri)

    with open(OUTPUT_FILE, "wb") as f:
        pickle.dump(model, f)

    logger.info(f"Model saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

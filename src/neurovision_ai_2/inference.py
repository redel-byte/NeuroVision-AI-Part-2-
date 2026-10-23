from functools import lru_cache
from pathlib import Path
import os
from urllib.error import URLError
from urllib.request import urlopen

import cv2
import numpy as np
import tensorflow as tf


CLASSES = ("glioma", "meningioma", "notumor", "pituitary")
MODEL_NAME = "NeuroVisionAI-MobileNetV2"
LOCAL_CHECKPOINT = Path(__file__).resolve().parents[2] / "models" / "best_mobilenetv2_finetuned.keras"
IMAGE_SIZE = (224, 224)
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def get_champion_version():
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    if tracking_uri.startswith(("http://", "https://")):
        try:
            # Check the server first so a broken SSH tunnel fails in 3 seconds.
            with urlopen(f"{tracking_uri.rstrip('/')}/health", timeout=3):
                pass
        except (OSError, URLError) as error:
            raise ConnectionError(
                f"MLflow is not reachable at {tracking_uri}. "
                "Start the MLflow server or reconnect the SSH tunnel."
            ) from error

    from mlflow.tracking import MlflowClient  # Import only after the server responds.

    client = MlflowClient(tracking_uri=tracking_uri)
    return client.get_model_version_by_alias(MODEL_NAME, "champion").version


@lru_cache(maxsize=2)
def _load_model_version(tracking_uri, version):
    import mlflow
    import mlflow.tensorflow

    mlflow.set_tracking_uri(tracking_uri)  # Uses the same server as the notebook.
    model_uri = f"models:/{MODEL_NAME}/{version}"
    return mlflow.tensorflow.load_model(model_uri)  # Loads the version selected by the alias.


def load_champion():
    version = get_champion_version()  # Read the current alias on each call.
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    return _load_model_version(tracking_uri, version)  # Cache by server and version.


@lru_cache(maxsize=1)
def load_local_checkpoint():
    if not LOCAL_CHECKPOINT.is_file():
        raise FileNotFoundError(f"Local checkpoint not found: {LOCAL_CHECKPOINT}")
    return tf.keras.models.load_model(LOCAL_CHECKPOINT, compile=False)


def predict_image(path, source="champion"):
    image_path = Path(path)
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if image_path.suffix.lower() not in EXTENSIONS:
        raise ValueError("Use a JPG, JPEG, PNG, or BMP image.")

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # OpenCV loads BGR; training used RGB.
    image = cv2.resize(image, IMAGE_SIZE, interpolation=cv2.INTER_LINEAR)
    image = image.astype(np.float32)
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)  # Scales pixels to [-1, 1].
    batch = np.expand_dims(image, axis=0)  # Adds one batch dimension: (1, 224, 224, 3).

    if source == "local":
        model = load_local_checkpoint()  # Offline demo: no MLflow server needed.
        version = None
    elif source == "champion":
        version = get_champion_version()  # Select the registered champion.
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
        model = _load_model_version(tracking_uri, version)
    else:
        raise ValueError("source must be 'local' or 'champion'.")

    scores = model.predict(batch, verbose=0)[0]  # One score per class.
    if len(scores) != len(CLASSES):
        raise ValueError("The champion model must return four class scores.")

    winner = int(np.argmax(scores))
    return {
        "label": CLASSES[winner],
        "confidence": float(scores[winner]),
        "model_source": source,  # Lets callers distinguish a demo from the champion.
        "model_version": version,
        "probabilities": {
            name: float(score) for name, score in zip(CLASSES, scores)
        },
    }

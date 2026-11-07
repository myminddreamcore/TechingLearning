
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
try:
    from dotenv import load_dotenv
    load_dotenv(BASE_DIR / ".env")
except ImportError:
    pass

MODEL_PATH = Path(os.getenv("MODEL_PATH", BASE_DIR / "model" / "model.tflite"))
LABELS_PATH = Path(os.getenv("LABELS_PATH", BASE_DIR / "model" / "labels.txt"))
BACKEND = os.getenv("MODEL_BACKEND", "auto").lower()
RESIZE_MODE = os.getenv("RESIZE_MODE", "stretch").lower()
MAX_UPLOAD_MB = float(os.getenv("MAX_UPLOAD_MB", "10"))
MIN_CONFIDENCE = float(os.getenv("MIN_CONFIDENCE", "0.6"))

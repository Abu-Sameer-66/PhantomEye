import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# Paths
DATA_DIR = BASE_DIR / "data"
VIDEOS_DIR = DATA_DIR / "videos"
GALLERY_DIR = DATA_DIR / "gallery"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"

# Detection settings
DETECTION_CONF = 0.45
DETECTION_MODEL = "yolov8n.pt"
DEVICE = "cpu"

# Tracking settings
TRACK_MAX_AGE = 30
TRACK_MIN_HITS = 3
TRACK_IOU_THRESH = 0.3

# Re-ID settings
REID_MODEL = "osnet_x0_25"
REID_THRESHOLD = 0.65

# API settings
API_HOST = "0.0.0.0"
API_PORT = 8000
SECRET_KEY = "phantomeye-secret-key-change-in-production"

# Analytics settings
HEATMAP_ALPHA = 0.6
DWELL_TIME_THRESHOLD = 30

print(f"[PhantomEye] Config loaded — Base: {BASE_DIR}")
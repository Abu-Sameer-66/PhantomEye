import os
import torch
import numpy as np
import cv2
from torchreid.reid.models import build_model

import warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "osnet_phantomeye_reid.pth")
DEVICE = "cpu"
IMG_HEIGHT = 256
IMG_WIDTH = 128


def load_reid_model():
    model = build_model(
        name="osnet_x0_25",
        num_classes=751,
        pretrained=False,
        use_gpu=False
    )
    state_dict = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def preprocess_crop(crop: np.ndarray) -> torch.Tensor:
    img = cv2.resize(crop, (IMG_WIDTH, IMG_HEIGHT))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = (img - mean) / std
    img = img.transpose(2, 0, 1)
    return torch.tensor(img, dtype=torch.float32).unsqueeze(0)


def extract_feature(model, crop: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        tensor = preprocess_crop(crop)
        feature = model(tensor)
        feature = feature.squeeze().numpy()
        feature = feature / (np.linalg.norm(feature) + 1e-8)
    return feature


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def match_person(query_crop: np.ndarray, gallery: list[dict], model, threshold: float = 0.6) -> dict:
    """
    Match query person crop against gallery.
    gallery = [{"id": int, "feature": np.ndarray, "crop": np.ndarray}]
    Returns best match or None.
    """
    if not gallery:
        return {"matched": False, "id": None, "similarity": 0.0}

    query_feat = extract_feature(model, query_crop)
    best_sim = -1
    best_id = None

    for entry in gallery:
        sim = cosine_similarity(query_feat, entry["feature"])
        if sim > best_sim:
            best_sim = sim
            best_id = entry["id"]

    if best_sim >= threshold:
        return {"matched": True, "id": best_id, "similarity": round(best_sim, 4)}
    else:
        return {"matched": False, "id": None, "similarity": round(best_sim, 4)}
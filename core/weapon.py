import os
import cv2
import numpy as np
from ultralytics import YOLO

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "weapon_detector.pt")

WEAPON_CLASSES = {
    0: "Handgun",
    1: "Knife",
    2: "Shotgun",
    3: "Sniper",
    4: "Automatic Rifle",
    5: "SMG",
    6: "Sword",
    7: "Bazooka",
    8: "Grenade Launcher"
}

THREAT_COLORS = {
    "Handgun":          (0, 0, 255),
    "Knife":            (0, 69, 255),
    "Shotgun":          (0, 0, 200),
    "Sniper":           (0, 0, 180),
    "Automatic Rifle":  (0, 0, 220),
    "SMG":              (0, 0, 210),
    "Sword":            (0, 165, 255),
    "Bazooka":          (0, 0, 150),
    "Grenade Launcher": (0, 0, 130),
}


def load_weapon_model():
    model = YOLO(MODEL_PATH)
    return model


def detect_weapons(frame: np.ndarray, model, conf_threshold: float = 0.35) -> tuple[np.ndarray, list[dict]]:
    """
    Detect weapons in frame.
    Returns annotated frame and list of detections.
    Each detection: {class_name, confidence, bbox: [x1,y1,x2,y2]}
    """
    results = model(frame, conf=conf_threshold, verbose=False)
    detections = []
    annotated = frame.copy()

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_name = WEAPON_CLASSES.get(cls_id, "Unknown")
            color = THREAT_COLORS.get(class_name, (0, 0, 255))

            # Draw box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Label background
            label = f"⚠ {class_name} {conf:.0%}"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(annotated, (x1, y1 - lh - 8), (x1 + lw, y1), color, -1)
            cv2.putText(annotated, label, (x1, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            detections.append({
                "class_name": class_name,
                "confidence": round(conf, 3),
                "bbox": [x1, y1, x2, y2]
            })

    return annotated, detections
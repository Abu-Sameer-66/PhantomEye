import cv2
import numpy as np
from deepface import DeepFace


# Suppress TF logs
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


EMOTION_COLORS = {
    "happy":     (0, 255, 128),
    "sad":       (255, 100, 50),
    "angry":     (0, 0, 255),
    "fear":      (180, 0, 255),
    "surprise":  (0, 220, 255),
    "disgust":   (0, 140, 60),
    "neutral":   (0, 255, 136),
}

FONT = cv2.FONT_HERSHEY_SIMPLEX


def analyze_faces(frame: np.ndarray) -> list[dict]:
    try:
        results = DeepFace.analyze(
            img_path=frame,
            actions=["age", "gender", "emotion"],
            detector_backend="opencv",
            enforce_detection=False,
            silent=True,
        )
        results = results if isinstance(results, list) else [results]

        h, w = frame.shape[:2]
        min_face = int(min(h, w) * 0.25)
        filtered = [
            r for r in results
            if r.get("region", {}).get("w", 0) >= min_face
            and r.get("region", {}).get("h", 0) >= min_face
        ]
        return filtered
    except Exception:
        return []

def draw_emotion_overlays(frame: np.ndarray, results: list[dict]) -> np.ndarray:
    """
    Draw age, gender, emotion overlay boxes on frame for each detected face.
    """
    for face in results:
        region = face.get("region", {})
        x = region.get("x", 0)
        y = region.get("y", 0)
        w = region.get("w", 0)
        h = region.get("h", 0)

        if w == 0 or h == 0:
            continue

        emotion = face.get("dominant_emotion", "neutral")
        age = int(face.get("age", 0))
        gender = face.get("dominant_gender", face.get("gender", "Unknown"))

        if isinstance(gender, dict):
            gender = max(gender, key=gender.get)

        color = EMOTION_COLORS.get(emotion, (0, 255, 136))

        # Face bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        # Top label — emotion
        label_emotion = f"{emotion.upper()}"
        cv2.rectangle(frame, (x, y - 22), (x + w, y), color, -1)
        cv2.putText(frame, label_emotion, (x + 4, y - 6),
                    FONT, 0.52, (0, 0, 0), 1, cv2.LINE_AA)

        # Bottom label — age + gender
        label_demo = f"Age:{age}  {gender}"
        cv2.rectangle(frame, (x, y + h), (x + w, y + h + 22), color, -1)
        cv2.putText(frame, label_demo, (x + 4, y + h + 15),
                    FONT, 0.52, (0, 0, 0), 1, cv2.LINE_AA)

    return frame


def process_frame_emotion(frame: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    """
    Main entry point — analyze + draw.
    Returns annotated frame and raw results list.
    """
    results = analyze_faces(frame)
    annotated = draw_emotion_overlays(frame.copy(), results)
    return annotated, results
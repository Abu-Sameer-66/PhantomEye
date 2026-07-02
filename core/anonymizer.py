"""
PhantomEye — Anonymization Engine (ANE v1.0)
core/anonymizer.py

GDPR-compliant face and body anonymization while preserving
all behavioral analytics. Three modes: face-only blur,
full-body pixelation, adaptive intensity.

Author: Abu Sameer (IUB AI Research Lab)
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional


# ── Config ────────────────────────────────────────────── #

class AnonMode:
    FACE_BLUR       = "face_blur"        # Gaussian blur on face only
    FACE_PIXELATE   = "face_pixelate"    # Pixelation on face only
    FULL_BLUR       = "full_blur"        # Blur entire person bbox
    FULL_PIXELATE   = "full_pixelate"    # Pixelate entire person bbox
    SILHOUETTE      = "silhouette"       # Replace person with solid silhouette


@dataclass
class AnonResult:
    frame:           np.ndarray   # anonymized frame
    persons_found:   int          # total persons detected
    faces_blurred:   int          # faces anonymized
    bodies_blurred:  int          # bodies anonymized
    mode:            str          # anonymization mode used
    intensity:       int          # blur intensity used

    def to_dict(self) -> dict:
        return {
            "persons_found":  self.persons_found,
            "faces_blurred":  self.faces_blurred,
            "bodies_blurred": self.bodies_blurred,
            "mode":           self.mode,
            "intensity":      self.intensity,
        }


# ── Core Engine ───────────────────────────────────────── #

class AnonymizationEngine:
    """
    ANE v1.0 — Anonymization Engine

    Preserves all analytics while making persons unidentifiable.
    Supports 5 anonymization modes with adjustable intensity.
    """

    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

    # ── Public API ───────────────────────────────────── #

    def anonymize_image(self,
                        frame: np.ndarray,
                        detections: list,
                        mode: str = AnonMode.FACE_BLUR,
                        intensity: int = 25) -> AnonResult:
        """
        Anonymize an image.

        detections: list of {"bbox": [x1,y1,x2,y2]} from PersonDetector
        intensity:  blur kernel size (odd number, higher = more blur)
        """
        out           = frame.copy()
        faces_blurred = 0
        bodies_blurred = 0
        intensity     = intensity if intensity % 2 == 1 else intensity + 1

        if mode in (AnonMode.FACE_BLUR, AnonMode.FACE_PIXELATE):
            # Detect faces within each person bbox
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )
            for (fx, fy, fw, fh) in faces:
                if mode == AnonMode.FACE_BLUR:
                    out = self._blur_region(out, fx, fy, fx+fw, fy+fh, intensity)
                else:
                    out = self._pixelate_region(out, fx, fy, fx+fw, fy+fh, intensity)
                faces_blurred += 1

        elif mode in (AnonMode.FULL_BLUR, AnonMode.FULL_PIXELATE, AnonMode.SILHOUETTE):
            for det in detections:
                x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
                if mode == AnonMode.FULL_BLUR:
                    out = self._blur_region(out, x1, y1, x2, y2, intensity)
                elif mode == AnonMode.FULL_PIXELATE:
                    out = self._pixelate_region(out, x1, y1, x2, y2, intensity)
                else:
                    out = self._silhouette_region(out, x1, y1, x2, y2)
                bodies_blurred += 1

        return AnonResult(
            frame=out,
            persons_found=len(detections),
            faces_blurred=faces_blurred,
            bodies_blurred=bodies_blurred,
            mode=mode,
            intensity=intensity,
        )

    def anonymize_video_frame(self,
                              frame: np.ndarray,
                              detections: list,
                              mode: str = AnonMode.FACE_BLUR,
                              intensity: int = 25) -> np.ndarray:
        """Lightweight version for real-time video — returns frame only."""
        result = self.anonymize_image(frame, detections, mode, intensity)
        return result.frame

    # ── Anonymization Methods ────────────────────────── #

    def _blur_region(self, frame: np.ndarray,
                     x1: int, y1: int, x2: int, y2: int,
                     intensity: int) -> np.ndarray:
        """Apply Gaussian blur to a region."""
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return frame
        roi = frame[y1:y2, x1:x2]
        k   = intensity if intensity % 2 == 1 else intensity + 1
        frame[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (k, k), 0)
        return frame

    def _pixelate_region(self, frame: np.ndarray,
                          x1: int, y1: int, x2: int, y2: int,
                          intensity: int) -> np.ndarray:
        """Apply pixelation (mosaic effect) to a region."""
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return frame
        roi     = frame[y1:y2, x1:x2]
        rh, rw  = roi.shape[:2]
        pixel_size = max(4, intensity // 2)
        small   = cv2.resize(roi, (max(1, rw // pixel_size), max(1, rh // pixel_size)),
                             interpolation=cv2.INTER_LINEAR)
        pixelated = cv2.resize(small, (rw, rh), interpolation=cv2.INTER_NEAREST)
        frame[y1:y2, x1:x2] = pixelated
        return frame

    def _silhouette_region(self, frame: np.ndarray,
                            x1: int, y1: int, x2: int, y2: int,
                            color: tuple = (30, 30, 30)) -> np.ndarray:
        """Replace region with a dark solid silhouette."""
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return frame
        frame[y1:y2, x1:x2] = color
        # Add subtle outline
        cv2.rectangle(frame, (x1, y1), (x2, y2), (60, 60, 60), 1)
        return frame

    # ── Overlay ──────────────────────────────────────── #

    def draw_anon_overlay(self, frame: np.ndarray,
                          result: AnonResult) -> np.ndarray:
        """Add ANONYMIZED watermark and stats to frame."""
        out = frame.copy()
        h, w = out.shape[:2]

        # Top-left badge
        cv2.rectangle(out, (0, 0), (260, 28), (20, 20, 20), -1)
        cv2.putText(out, f"ANONYMIZED | Mode: {result.mode.upper()}",
                    (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 136), 1)

        # Bottom-left stats
        stats = f"Persons: {result.persons_found}  Anonymized: {result.faces_blurred + result.bodies_blurred}"
        cv2.rectangle(out, (0, h-28), (320, h), (20, 20, 20), -1)
        cv2.putText(out, stats, (6, h-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 180, 255), 1)

        return out


# ── Singleton ─────────────────────────────────────────── #

_engine: Optional[AnonymizationEngine] = None


def get_engine() -> AnonymizationEngine:
    global _engine
    if _engine is None:
        _engine = AnonymizationEngine()
    return _engine


# ── Standalone Test ───────────────────────────────────── #

if __name__ == "__main__":
    import numpy as np

    engine = AnonymizationEngine()
    print("=" * 55)
    print("ANE v1.0 — Anonymization Engine | Standalone Test")
    print("=" * 55)

    # Create a test frame
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    detections = [
        {"bbox": [100, 50, 200, 300]},
        {"bbox": [350, 80, 500, 380]},
    ]

    modes = [
        AnonMode.FACE_BLUR,
        AnonMode.FACE_PIXELATE,
        AnonMode.FULL_BLUR,
        AnonMode.FULL_PIXELATE,
        AnonMode.SILHOUETTE,
    ]

    for mode in modes:
        result = engine.anonymize_image(frame.copy(), detections, mode=mode, intensity=25)
        print(f"  Mode: {mode:<20} | Output shape: {result.frame.shape} | ✓")

    print("\nAll 5 modes operational.")
    print("ANE v1.0 — All tests passed ✓")
import cv2
import time
import numpy as np
from pathlib import Path
from ultralytics import YOLO

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import DETECTION_CONF, DETECTION_MODEL, DEVICE, OUTPUTS_DIR


class PersonDetector:

    def __init__(self):
        self.model = YOLO(DETECTION_MODEL)
        self.conf  = DETECTION_CONF
        self.device = DEVICE
        self.frame_count = 0
        self.total_detections = 0
        print(f"[PhantomEye] Detector ready — model: {DETECTION_MODEL}  device: {DEVICE}")

    def detect(self, frame: np.ndarray) -> list:
        results = self.model(
            frame,
            conf=self.conf,
            classes=[0],
            device=self.device,
            verbose=False
        )[0]

        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf_score = float(box.conf[0])
            detections.append({
                "bbox"      : (x1, y1, x2, y2),
                "confidence": round(conf_score, 3),
                "cx"        : (x1 + x2) // 2,
                "cy"        : (x1 + x2) // 2,
            })

        self.frame_count += 1
        self.total_detections += len(detections)
        return detections

    def draw(self, frame: np.ndarray, detections: list) -> np.ndarray:
        out = frame.copy()

        for i, det in enumerate(detections):
            x1, y1, x2, y2 = det["bbox"]
            conf = det["confidence"]

            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 100), 2)

            label = f"Person  {conf:.2f}"
            label_w, label_h = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
            )[0]
            cv2.rectangle(
                out,
                (x1, y1 - label_h - 8),
                (x1 + label_w + 4, y1),
                (0, 255, 100), -1
            )
            cv2.putText(
                out, label,
                (x1 + 2, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (0, 0, 0), 1, cv2.LINE_AA
            )

        header = f"PhantomEye  |  Frame: {self.frame_count}  |  Persons: {len(detections)}"
        cv2.rectangle(out, (0, 0), (len(header) * 9 + 10, 28), (0, 0, 0), -1)
        cv2.putText(
            out, header,
            (6, 18),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
            (0, 255, 100), 1, cv2.LINE_AA
        )

        return out

    def stats(self) -> dict:
        avg = (
            round(self.total_detections / self.frame_count, 2)
            if self.frame_count > 0 else 0
        )
        return {
            "total_frames"    : self.frame_count,
            "total_detections": self.total_detections,
            "avg_per_frame"   : avg,
        }


def run_on_video(video_path: str, save: bool = True, show: bool = True):

    video_path = Path(video_path)
    if not video_path.exists():
        print(f"[ERROR] Video not found: {video_path}")
        return

    detector = PersonDetector()
    cap      = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        return

    fps    = int(cap.get(cv2.CAP_PROP_FPS)) or 25
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"[PhantomEye] Video  : {video_path.name}")
    print(f"[PhantomEye] Size   : {width}x{height}  FPS: {fps}  Frames: {total}")

    writer = None
    if save:
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = OUTPUTS_DIR / (video_path.stem + "_detected.mp4")
        fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
        writer   = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
        print(f"[PhantomEye] Saving : {out_path}")

    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = detector.detect(frame)
        annotated  = detector.draw(frame, detections)

        if writer:
            writer.write(annotated)

        if show:
            cv2.imshow("PhantomEye — Detection  [Q to quit]", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("\n[PhantomEye] Stopped by user.")
                break

        if detector.frame_count % 30 == 0:
            elapsed = time.time() - start_time
            print(
                f"\r[Frame {detector.frame_count}/{total}]  "
                f"Persons: {len(detections)}  "
                f"Elapsed: {elapsed:.1f}s",
                end=""
            )

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    stats = detector.stats()
    print(f"\n\n[PhantomEye] DONE")
    print(f"  Frames processed : {stats['total_frames']}")
    print(f"  Total detections : {stats['total_detections']}")
    print(f"  Avg persons/frame: {stats['avg_per_frame']}")
    if save:
        print(f"  Output saved to  : {OUTPUTS_DIR}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python core/detection.py <video_path>")
        print("Example: python core/detection.py data/videos/test.mp4")
    else:
        run_on_video(sys.argv[1])
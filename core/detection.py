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
        h, w = out.shape[:2]

        BLUE    = (255, 180, 0)   # brand accent blue  #00b4ff
        GREEN   = (136, 255, 0)   # brand accent green #00ff88
        CARD_BG = (32, 18, 6)     # brand card navy    #061220

        def blend_chip(x1, y1, x2, y2, alpha=0.88):
            roi = out[y1:y2, x1:x2]
            fill = np.empty_like(roi)
            fill[:, :] = CARD_BG
            cv2.addWeighted(fill, alpha, roi, 1 - alpha, 0, dst=roi)

        def overlaps(a, b):
            ax1, ay1, ax2, ay2 = a
            bx1, by1, bx2, by2 = b
            return not (ax2 < bx1 or bx2 < ax1 or ay2 < by1 or by2 < ay1)

        placed_labels = []

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            conf = det["confidence"]
            box_w, box_h = x2 - x1, y2 - y1

            arm = max(8, min(18, int(min(box_w, box_h) * 0.25)))
            for cx, cy, dx, dy in [
                (x1, y1, 1, 1), (x2, y1, -1, 1),
                (x1, y2, 1, -1), (x2, y2, -1, -1),
            ]:
                cv2.line(out, (cx, cy), (cx + dx * arm, cy), BLUE, 2, cv2.LINE_AA)
                cv2.line(out, (cx, cy), (cx, cy + dy * arm), BLUE, 2, cv2.LINE_AA)

            label = f"{conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.5, 1)
            pad_x, pad_y = 8, 6
            chip_w, chip_h = tw + pad_x * 2, th + pad_y * 2

            chip_x1 = min(x1 + 3, max(0, w - chip_w))
            chip_y1 = y1 + 3
            for _ in range(4):
                candidate = (chip_x1, chip_y1, chip_x1 + chip_w, chip_y1 + chip_h)
                if not any(overlaps(candidate, p) for p in placed_labels):
                    break
                chip_y1 += chip_h + 3
            chip_y1 = min(chip_y1, max(0, h - chip_h))
            chip_x2, chip_y2 = chip_x1 + chip_w, chip_y1 + chip_h
            placed_labels.append((chip_x1, chip_y1, chip_x2, chip_y2))

            blend_chip(chip_x1, chip_y1, chip_x2, chip_y2)
            cv2.rectangle(out, (chip_x1, chip_y1), (chip_x2, chip_y2), BLUE, 1, cv2.LINE_AA)
            cv2.putText(
                out, label,
                (chip_x1 + pad_x, chip_y2 - pad_y),
                cv2.FONT_HERSHEY_DUPLEX, 0.5, GREEN, 1, cv2.LINE_AA
            )

        header = f"PHANTOMEYE  |  FRAME {self.frame_count}  |  PERSONS {len(detections)}"
        (hdr_tw, hdr_th), _ = cv2.getTextSize(header, cv2.FONT_HERSHEY_DUPLEX, 0.5, 1)
        blend_chip(0, 0, hdr_tw + 20, hdr_th + 16)
        cv2.line(out, (0, hdr_th + 16), (hdr_tw + 20, hdr_th + 16), BLUE, 1, cv2.LINE_AA)
        cv2.putText(
            out, header,
            (10, hdr_th + 8),
            cv2.FONT_HERSHEY_DUPLEX, 0.5, GREEN, 1, cv2.LINE_AA
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
import cv2
import time
import numpy as np
from pathlib import Path
from collections import defaultdict

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from core.detection import PersonDetector
from config import OUTPUTS_DIR, TRACK_MAX_AGE, TRACK_IOU_THRESH


class Track:

    def __init__(self, track_id: int, bbox: tuple, conf: float):
        self.track_id   = track_id
        self.bbox       = bbox
        self.conf       = conf
        self.age        = 0
        self.hits       = 1
        self.lost       = 0
        self.trajectory = [self._center(bbox)]

    def _center(self, bbox: tuple) -> tuple:
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    def update(self, bbox: tuple, conf: float):
        self.bbox  = bbox
        self.conf  = conf
        self.hits += 1
        self.lost  = 0
        self.age  += 1
        cx, cy = self._center(bbox)
        self.trajectory.append((cx, cy))
        if len(self.trajectory) > 60:
            self.trajectory.pop(0)

    def mark_lost(self):
        self.lost += 1
        self.age  += 1

    def center(self) -> tuple:
        return self._center(self.bbox)


class ByteTracker:

    def __init__(self):
        self.tracks      = []
        self.next_id     = 1
        self.max_lost    = TRACK_MAX_AGE
        self.iou_thresh  = TRACK_IOU_THRESH
        self.frame_count = 0
        print("[PhantomEye] Tracker ready — ByteTracker (CPU-optimized)")

    def _iou(self, b1: tuple, b2: tuple) -> float:
        x1 = max(b1[0], b2[0])
        y1 = max(b1[1], b2[1])
        x2 = min(b1[2], b2[2])
        y2 = min(b1[3], b2[3])

        inter = max(0, x2 - x1) * max(0, y2 - y1)
        if inter == 0:
            return 0.0

        a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
        a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
        return inter / float(a1 + a2 - inter)

    def update(self, detections: list) -> list:
        self.frame_count += 1

        if not detections:
            for t in self.tracks:
                t.mark_lost()
            self.tracks = [t for t in self.tracks if t.lost <= self.max_lost]
            return []

        matched_track_ids = set()
        matched_det_ids   = set()

        for di, det in enumerate(detections):
            best_iou      = self.iou_thresh
            best_track_id = -1

            for ti, trk in enumerate(self.tracks):
                if ti in matched_track_ids:
                    continue
                iou_val = self._iou(det["bbox"], trk.bbox)
                if iou_val > best_iou:
                    best_iou      = iou_val
                    best_track_id = ti

            if best_track_id >= 0:
                self.tracks[best_track_id].update(det["bbox"], det["confidence"])
                matched_track_ids.add(best_track_id)
                matched_det_ids.add(di)

        for ti, trk in enumerate(self.tracks):
            if ti not in matched_track_ids:
                trk.mark_lost()

        for di, det in enumerate(detections):
            if di not in matched_det_ids:
                new_trk = Track(self.next_id, det["bbox"], det["confidence"])
                self.tracks.append(new_trk)
                self.next_id += 1

        self.tracks = [t for t in self.tracks if t.lost <= self.max_lost]

        return [t for t in self.tracks if t.lost == 0]

    def draw(self, frame: np.ndarray, active_tracks: list) -> np.ndarray:
        out = frame.copy()

        colors = [
            (0, 255, 100),  (0, 180, 255),  (255, 100, 0),
            (255, 0, 180),  (100, 255, 0),  (0, 100, 255),
            (255, 200, 0),  (0, 255, 200),  (200, 0, 255),
            (255, 50, 50),
        ]

        for trk in active_tracks:
            color = colors[trk.track_id % len(colors)]
            x1, y1, x2, y2 = trk.bbox

            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

            label = f"ID:{trk.track_id}  {trk.conf:.2f}"
            lw, lh = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
            )[0]
            cv2.rectangle(
                out,
                (x1, y1 - lh - 8),
                (x1 + lw + 4, y1),
                color, -1
            )
            cv2.putText(
                out, label,
                (x1 + 2, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (0, 0, 0), 1, cv2.LINE_AA
            )

            if len(trk.trajectory) > 1:
                for i in range(1, len(trk.trajectory)):
                    if trk.trajectory[i - 1] and trk.trajectory[i]:
                        cv2.line(
                            out,
                            trk.trajectory[i - 1],
                            trk.trajectory[i],
                            color, 1, cv2.LINE_AA
                        )

        header = (
            f"PhantomEye  |  Frame: {self.frame_count}"
            f"  |  Active: {len(active_tracks)}"
            f"  |  Total IDs: {self.next_id - 1}"
        )
        cv2.rectangle(out, (0, 0), (len(header) * 9 + 10, 28), (0, 0, 0), -1)
        cv2.putText(
            out, header, (6, 18),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
            (0, 255, 100), 1, cv2.LINE_AA
        )

        return out


def run_tracking(video_path: str, save: bool = True, show: bool = True):

    video_path = Path(video_path)
    if not video_path.exists():
        print(f"[ERROR] Video not found: {video_path}")
        return

    detector = PersonDetector()
    tracker  = ByteTracker()
    cap      = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"[ERROR] Cannot open: {video_path}")
        return

    fps    = int(cap.get(cv2.CAP_PROP_FPS)) or 25
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"[PhantomEye] {video_path.name} — {width}x{height} {fps}fps {total}frames")

    writer = None
    if save:
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = OUTPUTS_DIR / (video_path.stem + "_tracked.mp4")
        writer   = cv2.VideoWriter(
            str(out_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps, (width, height)
        )
        print(f"[PhantomEye] Saving to: {out_path}")

    start = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections     = detector.detect(frame)
        active_tracks  = tracker.update(detections)
        annotated      = tracker.draw(frame, active_tracks)

        if writer:
            writer.write(annotated)

        if show:
            cv2.imshow("PhantomEye — Tracking  [Q to quit]", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("\n[PhantomEye] Stopped.")
                break

        if tracker.frame_count % 30 == 0:
            print(
                f"\r[Frame {tracker.frame_count}/{total}]"
                f"  Active: {len(active_tracks)}"
                f"  Total IDs: {tracker.next_id - 1}"
                f"  Time: {time.time()-start:.1f}s",
                end=""
            )

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    print(f"\n[PhantomEye] Tracking done!")
    print(f"  Total frames  : {tracker.frame_count}")
    print(f"  Unique persons: {tracker.next_id - 1}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python core/tracker.py <video_path>")
    else:
        run_tracking(sys.argv[1])
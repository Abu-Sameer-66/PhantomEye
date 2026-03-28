import cv2
import time
import numpy as np
from pathlib import Path
from collections import defaultdict

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from core.detection import PersonDetector
from core.tracker import ByteTracker
from config import OUTPUTS_DIR, HEATMAP_ALPHA, DWELL_TIME_THRESHOLD


class BehavioralAnalyzer:

    def __init__(self, frame_width: int, frame_height: int, fps: int = 25):
        self.width      = frame_width
        self.height     = frame_height
        self.fps        = fps
        self.heatmap    = np.zeros((frame_height, frame_width), dtype=np.float32)
        self.dwell      = defaultdict(int)
        self.last_pos   = {}
        self.alerts     = []
        self.frame_num  = 0
        print("[PhantomEye] Behavioral analyzer ready")

    def update(self, active_tracks: list):
        self.frame_num += 1

        for trk in active_tracks:
            cx, cy = trk.center()

            cx = max(0, min(cx, self.width  - 1))
            cy = max(0, min(cy, self.height - 1))

            cv2.circle(
                self.heatmap,
                (cx, cy), 25, 1.0, -1
            )

            self.dwell[trk.track_id] += 1

            dwell_secs = self.dwell[trk.track_id] / self.fps
            if dwell_secs >= DWELL_TIME_THRESHOLD:
                already = any(
                    a["id"] == trk.track_id and a["type"] == "loitering"
                    for a in self.alerts
                )
                if not already:
                    self.alerts.append({
                        "type"     : "loitering",
                        "id"       : trk.track_id,
                        "bbox"     : trk.bbox,
                        "dwell_sec": round(dwell_secs, 1),
                        "frame"    : self.frame_num,
                    })

    def get_heatmap_overlay(self, frame: np.ndarray) -> np.ndarray:
        if self.heatmap.max() == 0:
            return frame

        normalized = cv2.normalize(
            self.heatmap, None, 0, 255, cv2.NORM_MINMAX
        )
        heat_uint8  = normalized.astype(np.uint8)
        heat_color  = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)
        blurred     = cv2.GaussianBlur(heat_color, (31, 31), 0)

        mask   = heat_uint8 > 10
        output = frame.copy()
        output[mask] = cv2.addWeighted(
            frame, 1 - HEATMAP_ALPHA,
            blurred, HEATMAP_ALPHA, 0
        )[mask]

        return output

    def draw_alerts(self, frame: np.ndarray, active_tracks: list) -> np.ndarray:
        out = frame.copy()

        active_ids = {t.track_id for t in active_tracks}
        recent = [
            a for a in self.alerts
            if a["id"] in active_ids and a["type"] == "loitering"
        ]

        for alert in recent:
            x1, y1, x2, y2 = alert["bbox"]
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 3)
            label = f"ALERT ID:{alert['id']} {alert['dwell_sec']}s"
            cv2.putText(
                out, label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 0, 255), 2, cv2.LINE_AA
            )

        return out

    def draw_dwell_info(self, frame: np.ndarray, active_tracks: list) -> np.ndarray:
        out = frame.copy()

        for trk in active_tracks:
            x1, y1, x2, y2 = trk.bbox
            secs = round(self.dwell[trk.track_id] / self.fps, 1)
            cv2.putText(
                out,
                f"{secs}s",
                (x1, y2 + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (255, 255, 0), 1, cv2.LINE_AA
            )

        return out

    def summary(self) -> dict:
        if not self.dwell:
            return {}

        dwell_secs = {
            k: round(v / self.fps, 1)
            for k, v in self.dwell.items()
        }
        return {
            "total_persons" : len(self.dwell),
            "total_alerts"  : len(self.alerts),
            "avg_dwell_sec" : round(
                sum(dwell_secs.values()) / len(dwell_secs), 2
            ),
            "max_dwell_sec" : max(dwell_secs.values()),
            "loiterers"     : [
                a["id"] for a in self.alerts
                if a["type"] == "loitering"
            ],
        }


def run_analytics(video_path: str, save: bool = True, show: bool = True):

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

    analyzer = BehavioralAnalyzer(width, height, fps)

    print(f"[PhantomEye] {video_path.name} — {width}x{height} {fps}fps")

    writer = None
    if save:
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = OUTPUTS_DIR / (video_path.stem + "_analytics.mp4")
        writer   = cv2.VideoWriter(
            str(out_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps, (width, height)
        )
        print(f"[PhantomEye] Saving to: {out_path}")

    show_heat  = False
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections    = detector.detect(frame)
        active_tracks = tracker.update(detections)
        analyzer.update(active_tracks)

        if show_heat:
            display = analyzer.get_heatmap_overlay(frame)
        else:
            display = frame.copy()

        display = tracker.draw(display, active_tracks)
        display = analyzer.draw_dwell_info(display, active_tracks)
        display = analyzer.draw_alerts(display, active_tracks)

        mode_text = "MODE: HEATMAP [H]" if show_heat else "MODE: TRACKING [H]"
        cv2.putText(
            display, mode_text,
            (width - 220, height - 12),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45,
            (200, 200, 200), 1, cv2.LINE_AA
        )

        if writer:
            writer.write(display)

        if show:
            cv2.imshow("PhantomEye — Analytics  [H=heatmap  Q=quit]", display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("h"):
                show_heat = not show_heat

        if tracker.frame_count % 30 == 0:
            print(
                f"\r[Frame {tracker.frame_count}/{total}]"
                f"  Active: {len(active_tracks)}"
                f"  Alerts: {len(analyzer.alerts)}"
                f"  Time: {time.time()-start_time:.1f}s",
                end=""
            )

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    summary = analyzer.summary()
    print(f"\n\n[PhantomEye] Analytics done!")
    print(f"  Total persons tracked : {summary.get('total_persons', 0)}")
    print(f"  Avg dwell time        : {summary.get('avg_dwell_sec', 0)}s")
    print(f"  Max dwell time        : {summary.get('max_dwell_sec', 0)}s")
    print(f"  Loitering alerts      : {summary.get('total_alerts', 0)}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python core/analytics.py <video_path>")
    else:
        run_analytics(sys.argv[1])
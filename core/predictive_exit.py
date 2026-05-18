"""
PhantomEye — Predictive Exit Vector (PEV v1.0)
Upgrade 10 | core/predictive_exit.py

Novel contribution: Frame-boundary exit prediction 3-5s before actual exit.
Uses velocity smoothing + linear trajectory extrapolation + boundary proximity weighting.
No equivalent open-source implementation exists for real-time multi-person surveillance.

Author: Abu Sameer (IUB AI Research Lab)
"""

import numpy as np
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Optional


# --- Config --------------------------------------------------------------- #

HISTORY_LEN       = 15          # frames of position history per person
SMOOTH_WINDOW     = 5           # velocity smoothing window
FPS_DEFAULT       = 25          # assumed FPS if not provided
PREDICT_SECONDS   = 4.0         # how many seconds ahead to project
BOUNDARY_MARGIN   = 40          # px — "near boundary" threshold
MIN_HISTORY       = 6           # minimum frames before prediction is valid
CONFIDENCE_DECAY  = 0.92        # per-frame decay on prediction confidence


# --- Data structures ------------------------------------------------------ #

@dataclass
class ExitPrediction:
    person_id: int
    exit_side: str                  # LEFT | RIGHT | TOP | BOTTOM | NONE
    seconds_to_exit: float          # estimated seconds remaining
    confidence: float               # 0.0 – 1.0
    predicted_exit_point: tuple     # (x, y) pixel coordinate on frame edge
    current_velocity: tuple         # (vx, vy) px/frame
    trajectory_points: list         # list of (x, y) for visualization
    alert: bool                     # True if exit imminent (< 2s)

    def to_dict(self) -> dict:
        return {
            "person_id":            self.person_id,
            "exit_side":            self.exit_side,
            "seconds_to_exit":      round(self.seconds_to_exit, 2),
            "confidence":           round(self.confidence, 3),
            "predicted_exit_point": list(self.predicted_exit_point),
            "current_velocity":     [round(v, 2) for v in self.current_velocity],
            "trajectory_points":    self.trajectory_points,
            "alert":                self.alert,
        }


@dataclass
class PersonTrack:
    person_id: int
    positions: deque = field(default_factory=lambda: deque(maxlen=HISTORY_LEN))
    frame_ids: deque = field(default_factory=lambda: deque(maxlen=HISTORY_LEN))


# --- Core engine ---------------------------------------------------------- #

class PredictiveExitEngine:
    """
    PEV v1.0 — Predictive Exit Vector Engine

    Algorithm overview:
    1. Maintain position history per tracked person.
    2. Compute smoothed velocity using a sliding window.
    3. Extrapolate linear trajectory forward PREDICT_SECONDS * FPS frames.
    4. Find first intersection of trajectory with frame boundaries.
    5. Compute confidence: velocity stability × boundary proximity × history depth.
    6. Emit ExitPrediction objects consumed by API + UI.
    """

    def __init__(self, frame_width: int = 640, frame_height: int = 480,
                 fps: float = FPS_DEFAULT):
        self.W   = frame_width
        self.H   = frame_height
        self.fps = fps
        self.tracks: dict[int, PersonTrack] = {}
        self.frame_counter = 0

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def update(self, detections: list[dict], frame_id: int = None) -> list[ExitPrediction]:
        """
        Call once per frame.

        detections: list of dicts with keys:
            - person_id (int)
            - bbox      (x1, y1, x2, y2)  — pixel coordinates

        Returns list of ExitPrediction for all tracked persons.
        """
        if frame_id is None:
            frame_id = self.frame_counter
        self.frame_counter += 1

        seen_ids = set()
        for det in detections:
            pid  = det["person_id"]
            bbox = det["bbox"]
            cx   = (bbox[0] + bbox[2]) / 2.0
            cy   = (bbox[1] + bbox[3]) / 2.0

            if pid not in self.tracks:
                self.tracks[pid] = PersonTrack(person_id=pid)

            self.tracks[pid].positions.append((cx, cy))
            self.tracks[pid].frame_ids.append(frame_id)
            seen_ids.add(pid)

        # remove stale tracks (not seen for > 60 frames)
        stale = [pid for pid in self.tracks
                 if pid not in seen_ids
                 and len(self.tracks[pid].frame_ids) > 0
                 and (frame_id - self.tracks[pid].frame_ids[-1]) > 60]
        for pid in stale:
            del self.tracks[pid]

        predictions = []
        for pid, track in self.tracks.items():
            pred = self._predict(track)
            if pred is not None:
                predictions.append(pred)

        return predictions

    def get_prediction(self, person_id: int) -> Optional[ExitPrediction]:
        """Get latest prediction for a single person."""
        track = self.tracks.get(person_id)
        if track is None:
            return None
        return self._predict(track)

    def reset(self):
        self.tracks.clear()
        self.frame_counter = 0

    # ------------------------------------------------------------------ #
    # Internal                                                             #
    # ------------------------------------------------------------------ #

    def _smooth_velocity(self, positions: list) -> tuple:
        """
        Sliding-window average of frame-to-frame displacement vectors.
        Returns (vx, vy) in px/frame.
        """
        if len(positions) < 2:
            return (0.0, 0.0)

        window = positions[-min(SMOOTH_WINDOW, len(positions)):]
        vx_list, vy_list = [], []
        for i in range(1, len(window)):
            vx_list.append(window[i][0] - window[i - 1][0])
            vy_list.append(window[i][1] - window[i - 1][1])

        return (float(np.mean(vx_list)), float(np.mean(vy_list)))

    def _velocity_stability(self, positions: list) -> float:
        """
        Returns 0–1. Higher = more consistent direction = higher confidence.
        Uses coefficient of variation of velocity magnitudes.
        """
        if len(positions) < 3:
            return 0.3

        magnitudes = []
        for i in range(1, len(positions)):
            dx = positions[i][0] - positions[i - 1][0]
            dy = positions[i][1] - positions[i - 1][1]
            magnitudes.append(np.sqrt(dx**2 + dy**2))

        mag_arr = np.array(magnitudes)
        if mag_arr.mean() < 1e-6:
            return 0.1   # nearly stationary — low confidence
        cv = mag_arr.std() / mag_arr.mean()
        return float(np.clip(1.0 - cv * 0.5, 0.1, 1.0))

    def _boundary_proximity_score(self, cx: float, cy: float) -> float:
        """
        Returns 0–1. Higher = closer to a frame edge.
        Used to boost confidence when person is already near boundary.
        """
        dist_left   = cx
        dist_right  = self.W - cx
        dist_top    = cy
        dist_bottom = self.H - cy

        min_dist = min(dist_left, dist_right, dist_top, dist_bottom)
        score = 1.0 - np.clip(min_dist / (max(self.W, self.H) * 0.5), 0.0, 1.0)
        return float(score)

    def _intersect_boundary(self, cx: float, cy: float,
                            vx: float, vy: float,
                            max_frames: int) -> tuple:
        """
        Walk the trajectory forward step by step.
        Returns (exit_side, frames_to_exit, exit_x, exit_y, trajectory_pts).
        Returns ('NONE', inf, cx, cy, []) if no exit within max_frames.
        """
        trajectory_pts = []
        step_x, step_y = cx, cy

        for f in range(1, max_frames + 1):
            step_x += vx
            step_y += vy

            if f % max(1, max_frames // 8) == 0:
                trajectory_pts.append((round(step_x, 1), round(step_y, 1)))

            if step_x <= 0:
                return ("LEFT",   f, 0,       step_y, trajectory_pts)
            if step_x >= self.W:
                return ("RIGHT",  f, self.W,   step_y, trajectory_pts)
            if step_y <= 0:
                return ("TOP",    f, step_x,   0,      trajectory_pts)
            if step_y >= self.H:
                return ("BOTTOM", f, step_x,   self.H, trajectory_pts)

        return ("NONE", float("inf"), step_x, step_y, trajectory_pts)

    def _predict(self, track: PersonTrack) -> Optional[ExitPrediction]:
        positions = list(track.positions)

        if len(positions) < MIN_HISTORY:
            return None

        cx, cy = positions[-1]
        vx, vy = self._smooth_velocity(positions)

        speed = np.sqrt(vx**2 + vy**2)
        if speed < 0.3:
            # essentially stationary — no meaningful prediction
            return ExitPrediction(
                person_id=track.person_id,
                exit_side="NONE",
                seconds_to_exit=float("inf"),
                confidence=0.0,
                predicted_exit_point=(round(cx), round(cy)),
                current_velocity=(round(vx, 2), round(vy, 2)),
                trajectory_points=[],
                alert=False,
            )

        max_frames = int(PREDICT_SECONDS * self.fps)
        exit_side, frames_out, ex, ey, traj = self._intersect_boundary(
            cx, cy, vx, vy, max_frames
        )

        seconds_to_exit = frames_out / self.fps if frames_out != float("inf") else float("inf")

        # confidence composition
        stab  = self._velocity_stability(positions)
        prox  = self._boundary_proximity_score(cx, cy)
        depth = np.clip(len(positions) / HISTORY_LEN, 0.0, 1.0)

        if exit_side == "NONE":
            confidence = 0.0
        else:
            confidence = float(
                np.clip(0.50 * stab + 0.30 * prox + 0.20 * depth, 0.0, 1.0)
            )
            # apply per-frame decay for far predictions
            confidence *= (CONFIDENCE_DECAY ** max(0, frames_out - int(self.fps)))

        alert = (exit_side != "NONE") and (seconds_to_exit <= 2.0) and (confidence >= 0.4)

        return ExitPrediction(
            person_id=track.person_id,
            exit_side=exit_side,
            seconds_to_exit=round(seconds_to_exit, 2),
            confidence=round(float(confidence), 3),
            predicted_exit_point=(round(ex), round(ey)),
            current_velocity=(round(vx, 2), round(vy, 2)),
            trajectory_points=traj,
            alert=alert,
        )


# --- Singleton accessor --------------------------------------------------- #

_engine: Optional[PredictiveExitEngine] = None


def get_engine(frame_width: int = 640, frame_height: int = 480,
               fps: float = FPS_DEFAULT) -> PredictiveExitEngine:
    global _engine
    if _engine is None:
        _engine = PredictiveExitEngine(frame_width, frame_height, fps)
    return _engine


def reset_engine():
    global _engine
    if _engine:
        _engine.reset()


# --- Standalone test ------------------------------------------------------- #

if __name__ == "__main__":
    engine = PredictiveExitEngine(frame_width=640, frame_height=480, fps=25)

    print("=" * 60)
    print("PEV v1.0 — Predictive Exit Vector | Standalone Test")
    print("=" * 60)

    # Simulate person moving toward right edge
    detections_sequence = []
    for i in range(20):
        x1 = 400 + i * 12
        y1 = 200
        detections_sequence.append([{"person_id": 1, "bbox": [x1, y1, x1 + 50, y1 + 100]}])

    for frame_idx, dets in enumerate(detections_sequence):
        preds = engine.update(dets, frame_id=frame_idx)
        for p in preds:
            status = "ALERT" if p.alert else "     "
            print(f"[Frame {frame_idx:02d}] [{status}] Person {p.person_id} | "
                  f"Exit: {p.exit_side:<7} | ETA: {p.seconds_to_exit:5.2f}s | "
                  f"Conf: {p.confidence:.3f} | "
                  f"ExitPt: {p.predicted_exit_point}")

    print("\nTest complete. PEV v1.0 operational.")
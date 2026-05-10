"""
PhantomEye — Threat Momentum Score (TMS)
=========================================
Novel Algorithm: Temporal threat accumulation across behavioral signals.
Unlike binary threat detection, TMS builds momentum over time —
compounding multiple weak signals into a strong, explainable threat score.

Research Contribution:
    "Temporal Threat Accumulation in Multi-Modal Surveillance Analysis"
    — No existing open-source system implements unified TMS.

Author: Abu Sameer (Abu-Sameer-66)
"""

import time
import numpy as np
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Optional


# ── THREAT LEVEL THRESHOLDS ──────────────────────────────────────
LEVEL_LOW      = 25.0
LEVEL_MEDIUM   = 50.0
LEVEL_HIGH     = 75.0
LEVEL_CRITICAL = 90.0

# ── SIGNAL WEIGHTS (tunable per deployment context) ───────────────
WEIGHTS = {
    "loitering":          0.28,   # staying too long in one zone
    "stress_emotion":     0.22,   # fear / angry / disgust detected
    "rapid_movement":     0.18,   # sudden velocity spike
    "proximity_violation":0.15,   # entering restricted zone
    "gaze_pattern":       0.10,   # erratic scanning behavior
    "group_anomaly":      0.07,   # unusual group formation
}

# ── DECAY CONSTANT ────────────────────────────────────────────────
# TMS decays over time if no new signals — like radioactive decay
DECAY_HALF_LIFE = 45.0   # seconds — score halves every 45s with no signals

# ── STRESS EMOTIONS ───────────────────────────────────────────────
STRESS_EMOTIONS = {"angry", "fear", "disgust", "sad"}


@dataclass
class SignalEvent:
    """A single behavioral signal captured at a point in time."""
    signal_type:  str
    strength:     float          # 0.0 → 1.0
    timestamp:    float = field(default_factory=time.time)
    metadata:     dict  = field(default_factory=dict)


@dataclass
class TMSResult:
    """Full TMS output for one person at one moment."""
    person_id:        int
    tms_score:        float        # 0–100
    threat_level:     str          # CLEAR / LOW / MEDIUM / HIGH / CRITICAL
    momentum:         float        # rate of change (rising/falling)
    active_signals:   list         # which signals contributed
    signal_breakdown: dict         # per-signal contribution
    time_in_system:   float        # seconds since first seen
    alert:            bool         # True if HIGH or CRITICAL
    alert_message:    str


class PersonThreatProfile:
    """
    Tracks evolving threat state for a single person across time.
    Core of the TMS algorithm — maintains rolling signal history.
    """

    def __init__(self, person_id: int):
        self.person_id       = person_id
        self.first_seen      = time.time()
        self.last_update     = time.time()
        self.tms_score       = 0.0
        self.signal_history  = deque(maxlen=200)   # rolling window
        self.position_history= deque(maxlen=100)
        self.velocity_history= deque(maxlen=50)
        self.emotion_history = deque(maxlen=30)
        self.dwell_seconds   = 0.0
        self.zone_entries    = defaultdict(int)
        self._prev_score     = 0.0

    # ── CORE UPDATE ──────────────────────────────────────────────
    def update(
        self,
        position:          Optional[tuple]  = None,
        emotion:           Optional[str]    = None,
        dwell_seconds:     float            = 0.0,
        is_loitering:      bool             = False,
        in_restricted_zone:bool            = False,
        group_anomaly:     bool             = False,
    ) -> "TMSResult":
        now = time.time()
        dt  = now - self.last_update
        self.last_update  = now
        self.dwell_seconds = dwell_seconds

        # ── 1. DECAY existing score ───────────────────────────────
        decay_factor  = 0.5 ** (dt / DECAY_HALF_LIFE)
        self.tms_score = self.tms_score * decay_factor

        # ── 2. COLLECT signals this frame ────────────────────────
        signals = []

        # Signal A — Loitering
        if is_loitering:
            strength = min(1.0, dwell_seconds / 120.0)   # max at 2 min
            signals.append(SignalEvent("loitering", strength,
                           metadata={"dwell_sec": dwell_seconds}))

        # Signal B — Stress emotion
        if emotion and emotion.lower() in STRESS_EMOTIONS:
            emotion_strength = {
                "fear": 0.95, "angry": 0.80,
                "disgust": 0.65, "sad": 0.45,
            }.get(emotion.lower(), 0.5)
            signals.append(SignalEvent("stress_emotion", emotion_strength,
                           metadata={"emotion": emotion}))
            self.emotion_history.append((now, emotion))

        # Signal C — Rapid movement (velocity spike)
        if position:
            self.position_history.append((now, position))
            velocity = self._compute_velocity()
            if velocity > 80:   # pixels/second threshold
                strength = min(1.0, (velocity - 80) / 200.0)
                signals.append(SignalEvent("rapid_movement", strength,
                               metadata={"velocity_px_s": round(velocity, 1)}))
            self.velocity_history.append(velocity)

        # Signal D — Proximity / restricted zone
        if in_restricted_zone:
            signals.append(SignalEvent("proximity_violation", 0.9,
                           metadata={"zone": "restricted"}))

        # Signal E — Gaze pattern (erratic position scanning)
        gaze_score = self._compute_gaze_anomaly()
        if gaze_score > 0.3:
            signals.append(SignalEvent("gaze_pattern", gaze_score,
                           metadata={"gaze_anomaly": round(gaze_score, 3)}))

        # Signal F — Group anomaly
        if group_anomaly:
            signals.append(SignalEvent("group_anomaly", 0.75))

        # ── 3. ACCUMULATE — compound interest model ───────────────
        signal_breakdown = {}
        for sig in signals:
            weight      = WEIGHTS.get(sig.signal_type, 0.1)
            contribution= sig.strength * weight * 100.0

            # Momentum amplifier: if score already high, new signals
            # contribute MORE — threat compounds like interest
            amplifier   = 1.0 + (self.tms_score / 200.0)
            contribution *= amplifier

            self.tms_score = min(100.0, self.tms_score + contribution)
            signal_breakdown[sig.signal_type] = round(contribution, 2)
            self.signal_history.append(sig)

        # ── 4. COMPUTE MOMENTUM (rate of change) ─────────────────
        momentum = self.tms_score - self._prev_score
        self._prev_score = self.tms_score

        # ── 5. DETERMINE THREAT LEVEL ────────────────────────────
        score    = self.tms_score
        level    = self._score_to_level(score)
        alert    = level in ("HIGH", "CRITICAL")
        msg      = self._build_alert_message(level, score, signal_breakdown)
        active   = [s.signal_type for s in signals]

        return TMSResult(
            person_id        = self.person_id,
            tms_score        = round(score, 2),
            threat_level     = level,
            momentum         = round(momentum, 3),
            active_signals   = active,
            signal_breakdown = signal_breakdown,
            time_in_system   = round(now - self.first_seen, 1),
            alert            = alert,
            alert_message    = msg,
        )

    # ── HELPERS ──────────────────────────────────────────────────
    def _compute_velocity(self) -> float:
        if len(self.position_history) < 2:
            return 0.0
        (t1, p1), (t2, p2) = self.position_history[-2], self.position_history[-1]
        dt = max(t2 - t1, 0.001)
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        return np.sqrt(dx**2 + dy**2) / dt

    def _compute_gaze_anomaly(self) -> float:
        """
        Detects erratic scanning: many rapid small direction changes.
        High gaze anomaly = nervous / surveillance-aware behavior.
        """
        if len(self.position_history) < 5:
            return 0.0
        recent = [p for _, p in list(self.position_history)[-10:]]
        if len(recent) < 3:
            return 0.0
        # Direction change count
        directions = []
        for i in range(1, len(recent)):
            dx = recent[i][0] - recent[i-1][0]
            dy = recent[i][1] - recent[i-1][1]
            angle = np.arctan2(dy, dx)
            directions.append(angle)
        changes = sum(
            1 for i in range(1, len(directions))
            if abs(directions[i] - directions[i-1]) > np.pi / 4
        )
        return min(1.0, changes / 6.0)

    @staticmethod
    def _score_to_level(score: float) -> str:
        if score >= LEVEL_CRITICAL: return "CRITICAL"
        if score >= LEVEL_HIGH:     return "HIGH"
        if score >= LEVEL_MEDIUM:   return "MEDIUM"
        if score >= LEVEL_LOW:      return "LOW"
        return "CLEAR"

    @staticmethod
    def _build_alert_message(level: str, score: float, breakdown: dict) -> str:
        if level == "CLEAR":
            return f"No threat indicators detected. TMS: {score:.1f}"
        top = sorted(breakdown.items(), key=lambda x: x[1], reverse=True)
        top_signal = top[0][0].replace("_", " ").title() if top else "unknown"
        msgs = {
            "LOW":      f"Minor behavioral anomaly. Primary signal: {top_signal}. TMS: {score:.1f}",
            "MEDIUM":   f"Elevated threat momentum. Dominant: {top_signal}. TMS: {score:.1f} — Monitor closely.",
            "HIGH":     f"⚠ HIGH THREAT — {top_signal} driving momentum. TMS: {score:.1f} — Immediate review.",
            "CRITICAL": f"🚨 CRITICAL — Multi-signal threat confirmed. TMS: {score:.1f} — Alert security.",
        }
        return msgs.get(level, "")


class ThreatMomentumEngine:
    """
    System-level engine managing TMS across all tracked persons.
    Drop-in addition to existing PhantomEye pipeline.
    """

    def __init__(self):
        self.profiles: dict[int, PersonThreatProfile] = {}
        self.alert_log: list[dict] = []

    def update_person(
        self,
        person_id:          int,
        position:           Optional[tuple]  = None,
        emotion:            Optional[str]    = None,
        dwell_seconds:      float            = 0.0,
        is_loitering:       bool             = False,
        in_restricted_zone: bool             = False,
        group_anomaly:      bool             = False,
    ) -> TMSResult:
        if person_id not in self.profiles:
            self.profiles[person_id] = PersonThreatProfile(person_id)

        result = self.profiles[person_id].update(
            position           = position,
            emotion            = emotion,
            dwell_seconds      = dwell_seconds,
            is_loitering       = is_loitering,
            in_restricted_zone = in_restricted_zone,
            group_anomaly      = group_anomaly,
        )

        if result.alert:
            self.alert_log.append({
                "timestamp":    time.time(),
                "person_id":    person_id,
                "tms_score":    result.tms_score,
                "threat_level": result.threat_level,
                "message":      result.alert_message,
            })

        return result

    def update_all(self, persons: list[dict]) -> list[TMSResult]:
        """
        Batch update — pass list of person dicts from tracker output.
        Each dict: {id, bbox, emotion, dwell_seconds, loitering}
        """
        results = []
        for p in persons:
            bbox = p.get("bbox")
            cx   = int((bbox[0] + bbox[2]) / 2) if bbox else None
            cy   = int((bbox[1] + bbox[3]) / 2) if bbox else None
            pos  = (cx, cy) if cx is not None else None

            r = self.update_person(
                person_id          = p.get("id", 0),
                position           = pos,
                emotion            = p.get("emotion"),
                dwell_seconds      = p.get("dwell_seconds", 0.0),
                is_loitering       = p.get("loitering", False),
                in_restricted_zone = p.get("in_restricted_zone", False),
                group_anomaly      = p.get("group_anomaly", False),
            )
            results.append(r)
        return results

    def get_highest_threat(self) -> Optional[TMSResult]:
        """Returns current highest-TMS person across all tracked."""
        if not self.profiles:
            return None
        best_id = max(self.profiles, key=lambda pid: self.profiles[pid].tms_score)
        return self.update_person(best_id)

    def summary(self) -> dict:
        """Session-level threat summary."""
        scores = {pid: p.tms_score for pid, p in self.profiles.items()}
        levels = {}
        for pid, p in self.profiles.items():
            lvl = PersonThreatProfile._score_to_level(p.tms_score)
            levels[lvl] = levels.get(lvl, 0) + 1

        return {
            "total_persons_tracked": len(self.profiles),
            "total_alerts":          len(self.alert_log),
            "level_distribution":    levels,
            "highest_tms":           round(max(scores.values(), default=0), 2),
            "highest_tms_person":    max(scores, key=scores.get, default=None),
            "avg_tms":               round(np.mean(list(scores.values())), 2) if scores else 0.0,
            "recent_alerts":         self.alert_log[-5:],
        }

    def reset_person(self, person_id: int):
        """Clear threat profile — e.g. after false positive confirmed."""
        self.profiles.pop(person_id, None)

    def reset_all(self):
        self.profiles.clear()
        self.alert_log.clear()


# ── STANDALONE TEST ───────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("THREAT MOMENTUM SCORE — PhantomEye Novel Algorithm")
    print("=" * 60)

    engine = ThreatMomentumEngine()

    # Simulate person 1 — gradually escalating threat
    print("\n[SIMULATION] Person 1 — Escalating threat scenario")
    scenarios = [
        dict(person_id=1, position=(100, 200), dwell_seconds=10),
        dict(person_id=1, position=(105, 205), dwell_seconds=30, is_loitering=True),
        dict(person_id=1, position=(108, 210), dwell_seconds=60, is_loitering=True, emotion="fear"),
        dict(person_id=1, position=(300, 400), dwell_seconds=65, is_loitering=True, emotion="angry"),
        dict(person_id=1, position=(310, 405), dwell_seconds=70, is_loitering=True, emotion="angry", in_restricted_zone=True),
    ]

    for i, s in enumerate(scenarios):
        result = engine.update_person(**s)
        print(f"\n  Frame {i+1}: TMS={result.tms_score:.1f} | Level={result.threat_level} | Momentum={result.momentum:+.2f}")
        print(f"  Signals: {result.active_signals}")
        if result.alert:
            print(f"ALERT: {result.alert_message}")

    # Simulate person 2 — benign
    print("\n[SIMULATION] Person 2 — Normal behavior")
    r2 = engine.update_person(person_id=2, position=(500, 300), dwell_seconds=5)
    print(f"  TMS={r2.tms_score:.1f} | Level={r2.threat_level}")

    print("\n[SESSION SUMMARY]")
    s = engine.summary()
    for k, v in s.items():
        print(f"  {k}: {v}")
    print("=" * 60)
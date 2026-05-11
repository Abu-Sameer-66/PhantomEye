"""
PhantomEye — Behavioral DNA Fingerprint (BDF)
==============================================
Novel Algorithm: Camera-agnostic person identification via behavioral signature.
Identifies the same person across cameras using movement patterns alone —
no face, no biometrics required. Works through masks, hats, distance.

Research Contribution:
    "Camera-Agnostic Re-Identification via Behavioral Signature Synthesis"
    BDF = f(gait, velocity_profile, spatial_preference, social_distance, dwell_zones)

Author: Abu Sameer (Abu-Sameer-66)
"""

import time
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Optional


# Cosine similarity threshold for declaring a match
MATCH_THRESHOLD = 0.82

# Minimum observations before BDF vector is considered reliable
MIN_OBSERVATIONS = 15


@dataclass
class BDFVector:
    """
    The behavioral fingerprint for one person.
    A fixed-length feature vector derived purely from movement behavior.
    """
    person_id:           int
    gait_signature:      np.ndarray    # stride rhythm and step pattern
    velocity_profile:    np.ndarray    # speed distribution over time
    spatial_preference:  np.ndarray    # normalized heatmap of visited zones
    social_distance_avg: float         # average distance maintained from others
    dwell_zone_signature:np.ndarray    # which areas person stops in
    observation_count:   int           # how many frames contributed
    created_at:          float         # unix timestamp
    confidence:          float         # 0.0 to 1.0 — reliability of this BDF

    def to_flat(self) -> np.ndarray:
        """Flatten all components into a single feature vector for similarity computation."""
        return np.concatenate([
            self.gait_signature,
            self.velocity_profile,
            self.spatial_preference.flatten(),
            np.array([self.social_distance_avg]),
            self.dwell_zone_signature,
        ])


@dataclass
class BDFMatchResult:
    """Result of matching one person against the BDF gallery."""
    query_id:         int
    matched_id:       Optional[int]
    similarity:       float
    is_match:         bool
    confidence:       float
    method:           str            # "behavioral_dna"
    explanation:      str


class PersonBehaviorBuffer:
    """
    Collects raw observations for one tracked person and extracts BDF components.
    Acts as the feature extractor — raw trajectory → behavioral signature.
    """

    GRID_W = 8    # spatial grid width (normalized)
    GRID_H = 8    # spatial grid height

    def __init__(self, person_id: int, frame_w: int = 640, frame_h: int = 480):
        self.person_id    = person_id
        self.frame_w      = frame_w
        self.frame_h      = frame_h
        self.first_seen   = time.time()

        self.positions    = deque(maxlen=300)
        self.timestamps   = deque(maxlen=300)
        self.velocities   = deque(maxlen=300)
        self.step_lengths = deque(maxlen=150)    # gait
        self.dwell_events = deque(maxlen=50)     # positions where person stopped
        self.social_dists = deque(maxlen=100)    # distances from nearest other person

        # Spatial grid accumulator
        self.spatial_grid = np.zeros((self.GRID_H, self.GRID_W), dtype=np.float32)

    def observe(self, position: tuple, nearest_person_dist: float = 0.0):
        """Add one frame observation."""
        now = time.time()
        cx, cy = position

        self.positions.append((cx, cy))
        self.timestamps.append(now)
        self.social_dists.append(nearest_person_dist)

        # Update spatial grid
        gx = min(int((cx / self.frame_w) * self.GRID_W), self.GRID_W - 1)
        gy = min(int((cy / self.frame_h) * self.GRID_H), self.GRID_H - 1)
        self.spatial_grid[gy, gx] += 1.0

        # Compute velocity and gait from consecutive positions
        if len(self.positions) >= 2:
            prev = self.positions[-2]
            dt   = max(now - list(self.timestamps)[-2], 0.001)
            dx   = cx - prev[0]
            dy   = cy - prev[1]
            dist = np.sqrt(dx**2 + dy**2)
            vel  = dist / dt
            self.velocities.append(vel)
            self.step_lengths.append(dist)

            # Mark dwell if velocity drops below threshold
            if vel < 8.0:
                self.dwell_events.append((cx, cy))

    def is_ready(self) -> bool:
        return len(self.positions) >= MIN_OBSERVATIONS

    def extract_bdf(self) -> Optional[BDFVector]:
        """
        Extract the Behavioral DNA vector from accumulated observations.
        Returns None if insufficient data.
        """
        if not self.is_ready():
            return None

        # Gait signature — histogram of step lengths (captures stride rhythm)
        steps = np.array(self.step_lengths) if self.step_lengths else np.zeros(10)
        gait_sig = np.histogram(steps, bins=10, range=(0, 50))[0].astype(np.float32)
        if gait_sig.sum() > 0:
            gait_sig = gait_sig / gait_sig.sum()

        # Velocity profile — distribution of movement speeds
        vels = np.array(self.velocities) if self.velocities else np.zeros(10)
        vel_profile = np.histogram(vels, bins=10, range=(0, 200))[0].astype(np.float32)
        if vel_profile.sum() > 0:
            vel_profile = vel_profile / vel_profile.sum()

        # Spatial preference — normalized visit frequency per grid cell
        spatial = self.spatial_grid.copy()
        if spatial.sum() > 0:
            spatial = spatial / spatial.sum()

        # Social distance average
        social_avg = float(np.mean(self.social_dists)) if self.social_dists else 0.0

        # Dwell zone signature — where does this person stop
        dwell_grid = np.zeros((self.GRID_H, self.GRID_W), dtype=np.float32)
        for dx, dy in self.dwell_events:
            gx = min(int((dx / self.frame_w) * self.GRID_W), self.GRID_W - 1)
            gy = min(int((dy / self.frame_h) * self.GRID_H), self.GRID_H - 1)
            dwell_grid[gy, gx] += 1.0
        dwell_flat = dwell_grid.flatten()
        if dwell_flat.sum() > 0:
            dwell_flat = dwell_flat / dwell_flat.sum()

        # Confidence grows with observation count
        confidence = min(1.0, len(self.positions) / 100.0)

        return BDFVector(
            person_id            = self.person_id,
            gait_signature       = gait_sig,
            velocity_profile     = vel_profile,
            spatial_preference   = spatial,
            social_distance_avg  = social_avg,
            dwell_zone_signature = dwell_flat,
            observation_count    = len(self.positions),
            created_at           = self.first_seen,
            confidence           = confidence,
        )


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two flat feature vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-8 or norm_b < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


class BehavioralDNAEngine:
    """
    System-level engine managing BDF extraction and gallery matching.
    Drop-in addition to the PhantomEye pipeline alongside TMS.

    Workflow:
        1. Feed positions per tracked person each frame
        2. Once buffer is ready, extract BDF vector
        3. Match against gallery to detect re-entries / cross-camera IDs
    """

    def __init__(self, frame_w: int = 640, frame_h: int = 480):
        self.frame_w  = frame_w
        self.frame_h  = frame_h
        self.buffers: dict[int, PersonBehaviorBuffer] = {}
        self.gallery: dict[int, BDFVector]            = {}
        self.match_log: list[dict]                    = []

    def observe(self, person_id: int, position: tuple, nearest_dist: float = 0.0):
        """Feed one frame observation for a tracked person."""
        if person_id not in self.buffers:
            self.buffers[person_id] = PersonBehaviorBuffer(
                person_id, self.frame_w, self.frame_h
            )
        self.buffers[person_id].observe(position, nearest_dist)

    def extract_and_register(self, person_id: int) -> Optional[BDFVector]:
        """
        Extract BDF for a person and add to gallery.
        Call this when a person exits the scene or after sufficient observations.
        """
        buf = self.buffers.get(person_id)
        if buf is None or not buf.is_ready():
            return None

        bdf = buf.extract_bdf()
        if bdf:
            self.gallery[person_id] = bdf
        return bdf

    def match_against_gallery(self, person_id: int) -> BDFMatchResult:
        """
        Match person's current BDF against all gallery entries.
        Used to detect if a person seen before has re-entered (possibly different tracked ID).
        """
        buf = self.buffers.get(person_id)
        if buf is None or not buf.is_ready():
            return BDFMatchResult(
                query_id   = person_id,
                matched_id = None,
                similarity = 0.0,
                is_match   = False,
                confidence = 0.0,
                method     = "behavioral_dna",
                explanation = "Insufficient observations to build BDF."
            )

        query_bdf = buf.extract_bdf()
        if query_bdf is None:
            return BDFMatchResult(
                query_id   = person_id,
                matched_id = None,
                similarity = 0.0,
                is_match   = False,
                confidence = 0.0,
                method     = "behavioral_dna",
                explanation = "BDF extraction failed."
            )

        query_vec = query_bdf.to_flat()
        best_sim  = 0.0
        best_id   = None

        for gal_id, gal_bdf in self.gallery.items():
            if gal_id == person_id:
                continue
            sim = cosine_similarity(query_vec, gal_bdf.to_flat())
            if sim > best_sim:
                best_sim = sim
                best_id  = gal_id

        is_match = best_sim >= MATCH_THRESHOLD
        confidence = query_bdf.confidence

        if is_match:
            explanation = (
                f"Person {person_id} behavioral signature matches Person {best_id} "
                f"with {best_sim:.1%} similarity. Same individual likely re-entered scene."
            )
            self.match_log.append({
                "timestamp":  time.time(),
                "query_id":   person_id,
                "matched_id": best_id,
                "similarity": round(best_sim, 4),
            })
        else:
            explanation = (
                f"No gallery match above threshold ({MATCH_THRESHOLD:.0%}). "
                f"Best similarity: {best_sim:.1%}. Person appears to be new to scene."
            )

        return BDFMatchResult(
            query_id   = person_id,
            matched_id = best_id if is_match else None,
            similarity = round(best_sim, 4),
            is_match   = is_match,
            confidence = round(confidence, 3),
            method     = "behavioral_dna",
            explanation = explanation,
        )

    def get_bdf(self, person_id: int) -> Optional[BDFVector]:
        """Get current BDF vector for a person if ready."""
        buf = self.buffers.get(person_id)
        if buf and buf.is_ready():
            return buf.extract_bdf()
        return None

    def summary(self) -> dict:
        """Session summary of BDF engine state."""
        ready_count = sum(1 for b in self.buffers.values() if b.is_ready())
        return {
            "persons_tracked":    len(self.buffers),
            "bdf_ready":          ready_count,
            "gallery_size":       len(self.gallery),
            "matches_detected":   len(self.match_log),
            "recent_matches":     self.match_log[-3:],
        }

    def reset_person(self, person_id: int):
        self.buffers.pop(person_id, None)
        self.gallery.pop(person_id, None)

    def reset_all(self):
        self.buffers.clear()
        self.gallery.clear()
        self.match_log.clear()


if __name__ == "__main__":
    print("=" * 60)
    print("Behavioral DNA Fingerprint — PhantomEye")
    print("=" * 60)

    engine = BehavioralDNAEngine(frame_w=640, frame_h=480)

    # Simulate Person 1 — slow walker, prefers left side of frame
    print("\n[SIM] Person 1 — slow walker, left-side preference")
    for i in range(60):
        x = int(80 + i * 2.5 + np.random.randn() * 3)
        y = int(200 + np.sin(i * 0.3) * 20 + np.random.randn() * 2)
        engine.observe(1, (x, y), nearest_dist=120.0)

    # Register Person 1 in gallery
    bdf1 = engine.extract_and_register(1)
    if bdf1:
        print(f"  BDF extracted — confidence: {bdf1.confidence:.2f} | observations: {bdf1.observation_count}")

    # Simulate Person 2 — fast mover, center of frame
    print("\n[SIM] Person 2 — fast mover, center preference")
    for i in range(60):
        x = int(300 + i * 4 + np.random.randn() * 5)
        y = int(240 + np.random.randn() * 15)
        engine.observe(2, (x, y), nearest_dist=50.0)

    bdf2 = engine.extract_and_register(2)
    if bdf2:
        print(f"  BDF extracted — confidence: {bdf2.confidence:.2f} | observations: {bdf2.observation_count}")

    # Simulate Person 3 — same behavioral pattern as Person 1 (re-entry)
    print("\n[SIM] Person 3 — same as Person 1, re-entered scene with new tracking ID")
    for i in range(40):
        x = int(85 + i * 2.4 + np.random.randn() * 4)
        y = int(205 + np.sin(i * 0.3) * 18 + np.random.randn() * 3)
        engine.observe(3, (x, y), nearest_dist=115.0)

    result = engine.match_against_gallery(3)
    print(f"\n  Match result:")
    print(f"    Is match:   {result.is_match}")
    print(f"    Matched ID: {result.matched_id}")
    print(f"    Similarity: {result.similarity:.1%}")
    print(f"    Confidence: {result.confidence:.2f}")
    print(f"    {result.explanation}")

    print("\n[SESSION SUMMARY]")
    s = engine.summary()
    for k, v in s.items():
        print(f"  {k}: {v}")
    print("=" * 60)
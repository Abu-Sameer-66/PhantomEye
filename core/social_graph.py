"""
PhantomEye — Social Graph Intelligence (SGI)
=============================================
Novel Algorithm: Real-time group detection and relationship mapping
from surveillance footage — no prior information required.

Detects "who is with whom" purely from movement correlation,
shared dwell zones, and proximity patterns.

Research Contribution:
    "Implicit Social Graph Extraction from Behavioral Correlation
     in Multi-Person Surveillance Scenes"

Real-world application: Three bank robbers enter separately but
act as a coordinated group — SGI detects the association before
any overt action occurs.

Author: Abu Sameer (Abu-Sameer-66)
"""

import time
import numpy as np
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Optional


# Thresholds
PROXIMITY_THRESHOLD_PX  = 150    # pixels — max distance to consider "near"
VELOCITY_CORR_THRESHOLD = 0.70   # minimum correlation to flag movement sync
DWELL_OVERLAP_THRESHOLD = 0.60   # zone overlap ratio to flag shared stopping
MIN_FRAMES_FOR_LINK     = 20     # frames before declaring association


@dataclass
class SocialLink:
    """A detected social relationship between two tracked persons."""
    person_a:       int
    person_b:       int
    strength:       float          # 0.0 to 1.0
    link_type:      str            # "proximate" | "synchronized" | "coordinated"
    evidence:       dict           # what signals contributed
    first_detected: float          # unix timestamp
    frame_count:    int            # how many frames this link was observed


@dataclass
class GroupDetection:
    """A detected group — two or more associated persons."""
    group_id:     int
    members:      list[int]        # person IDs
    cohesion:     float            # average link strength within group
    formation:    str              # "pair" | "trio" | "cluster"
    alert:        bool             # True if suspicious
    alert_reason: str


class PersonSocialBuffer:
    """Tracks movement and position history for one person."""

    def __init__(self, person_id: int):
        self.person_id       = person_id
        self.positions       = deque(maxlen=150)
        self.timestamps      = deque(maxlen=150)
        self.velocities      = deque(maxlen=100)
        self.dwell_positions = deque(maxlen=60)
        self.first_seen      = time.time()

    def observe(self, position: tuple):
        now = time.time()
        cx, cy = position
        self.positions.append((cx, cy))
        self.timestamps.append(now)

        if len(self.positions) >= 2:
            prev = self.positions[-2]
            dt   = max(now - list(self.timestamps)[-2], 0.001)
            dx   = cx - prev[0]
            dy   = cy - prev[1]
            vel  = np.sqrt(dx**2 + dy**2) / dt
            self.velocities.append(vel)
            if vel < 8.0:
                self.dwell_positions.append((cx, cy))

    def current_position(self) -> Optional[tuple]:
        return self.positions[-1] if self.positions else None

    def velocity_series(self) -> np.ndarray:
        return np.array(self.velocities) if self.velocities else np.zeros(5)

    def is_ready(self) -> bool:
        return len(self.positions) >= 10


def euclidean(a: tuple, b: tuple) -> float:
    return np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)


def velocity_correlation(v1: np.ndarray, v2: np.ndarray) -> float:
    """Pearson correlation between two velocity series."""
    min_len = min(len(v1), len(v2), 30)
    if min_len < 5:
        return 0.0
    a = v1[-min_len:]
    b = v2[-min_len:]
    if np.std(a) < 1e-6 or np.std(b) < 1e-6:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def dwell_zone_overlap(dw1: list, dw2: list, radius: float = 60.0) -> float:
    """
    Fraction of person A's dwell positions that are within radius
    of any of person B's dwell positions.
    """
    if not dw1 or not dw2:
        return 0.0
    overlap = sum(
        1 for p in dw1
        if any(euclidean(p, q) < radius for q in dw2)
    )
    return overlap / len(dw1)


class SocialGraphEngine:
    """
    System-level engine that builds and maintains a real-time social graph
    from tracked person observations.

    Usage:
        engine = SocialGraphEngine()
        # each frame, for each tracked person:
        engine.observe(person_id, (cx, cy))
        # get current groups:
        groups = engine.detect_groups()
    """

    def __init__(self, proximity_px: float = PROXIMITY_THRESHOLD_PX):
        self.proximity_px   = proximity_px
        self.buffers:  dict[int, PersonSocialBuffer]       = {}
        self.links:    dict[tuple, SocialLink]             = {}
        self.frame_count = 0
        self.alert_log: list[dict] = []

    def observe(self, person_id: int, position: tuple):
        """Feed one frame observation for a tracked person."""
        if person_id not in self.buffers:
            self.buffers[person_id] = PersonSocialBuffer(person_id)
        self.buffers[person_id].observe(position)

    def observe_all(self, persons: list[dict]):
        """
        Batch update from tracker output.
        Each dict: {id, bbox} where bbox = [x1, y1, x2, y2]
        """
        self.frame_count += 1
        for p in persons:
            pid  = p.get("id", 0)
            bbox = p.get("bbox", [0, 0, 1, 1])
            cx   = int((bbox[0] + bbox[2]) / 2)
            cy   = int((bbox[1] + bbox[3]) / 2)
            self.observe(pid, (cx, cy))

        if self.frame_count % 10 == 0:
            self._update_links()

    def _update_links(self):
        """Recompute all pairwise social links."""
        ids = [pid for pid, buf in self.buffers.items() if buf.is_ready()]

        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a_id = ids[i]
                b_id = ids[j]
                self._compute_link(a_id, b_id)

    def _compute_link(self, a_id: int, b_id: int):
        buf_a = self.buffers[a_id]
        buf_b = self.buffers[b_id]

        pos_a = buf_a.current_position()
        pos_b = buf_b.current_position()
        if pos_a is None or pos_b is None:
            return

        dist = euclidean(pos_a, pos_b)
        vel_corr = velocity_correlation(
            buf_a.velocity_series(),
            buf_b.velocity_series()
        )
        dwell_overlap = dwell_zone_overlap(
            list(buf_a.dwell_positions),
            list(buf_b.dwell_positions)
        )

        # Proximity score — inverse of distance, normalized
        prox_score = max(0.0, 1.0 - (dist / (self.proximity_px * 3)))

        # Weighted link strength
        strength = (
            prox_score    * 0.40 +
            max(0, vel_corr) * 0.35 +
            dwell_overlap * 0.25
        )

        if strength < 0.15:
            # Remove weak link if it existed
            self.links.pop((a_id, b_id), None)
            return

        # Determine link type
        if vel_corr > VELOCITY_CORR_THRESHOLD and dwell_overlap > DWELL_OVERLAP_THRESHOLD:
            link_type = "coordinated"
        elif vel_corr > VELOCITY_CORR_THRESHOLD:
            link_type = "synchronized"
        else:
            link_type = "proximate"

        key = (min(a_id, b_id), max(a_id, b_id))

        if key in self.links:
            existing = self.links[key]
            # Smooth strength over time
            smoothed = existing.strength * 0.7 + strength * 0.3
            self.links[key] = SocialLink(
                person_a       = a_id,
                person_b       = b_id,
                strength       = round(smoothed, 4),
                link_type      = link_type,
                evidence       = {
                    "proximity_px":    round(dist, 1),
                    "velocity_corr":   round(vel_corr, 3),
                    "dwell_overlap":   round(dwell_overlap, 3),
                    "prox_score":      round(prox_score, 3),
                },
                first_detected = existing.first_detected,
                frame_count    = existing.frame_count + 1,
            )
        else:
            self.links[key] = SocialLink(
                person_a       = a_id,
                person_b       = b_id,
                strength       = round(strength, 4),
                link_type      = link_type,
                evidence       = {
                    "proximity_px":    round(dist, 1),
                    "velocity_corr":   round(vel_corr, 3),
                    "dwell_overlap":   round(dwell_overlap, 3),
                    "prox_score":      round(prox_score, 3),
                },
                first_detected = time.time(),
                frame_count    = 1,
            )

    def detect_groups(self) -> list[GroupDetection]:
        """
        Run connected-component analysis on the link graph
        to extract groups of associated persons.
        """
        # Build adjacency from confirmed links
        adjacency: dict[int, set] = defaultdict(set)
        for (a, b), link in self.links.items():
            if link.strength >= 0.30 and link.frame_count >= MIN_FRAMES_FOR_LINK:
                adjacency[a].add(b)
                adjacency[b].add(a)

        # BFS connected components
        visited = set()
        components = []
        for start in adjacency:
            if start in visited:
                continue
            component = set()
            queue = [start]
            while queue:
                node = queue.pop()
                if node in visited:
                    continue
                visited.add(node)
                component.add(node)
                queue.extend(adjacency[node] - visited)
            if len(component) >= 2:
                components.append(sorted(component))

        groups = []
        for gid, members in enumerate(components):
            # Average cohesion across all internal links
            internal_links = [
                self.links[(min(a, b), max(a, b))]
                for a in members for b in members
                if a < b and (min(a, b), max(a, b)) in self.links
            ]
            cohesion = float(np.mean([l.strength for l in internal_links])) if internal_links else 0.0

            n = len(members)
            formation = "pair" if n == 2 else "trio" if n == 3 else "cluster"

            # Alert if coordinated link exists within group
            has_coordinated = any(l.link_type == "coordinated" for l in internal_links)
            alert = has_coordinated and n >= 2
            alert_reason = (
                f"Coordinated group of {n} detected — synchronized movement and shared dwell zones."
                if alert else ""
            )

            if alert:
                self.alert_log.append({
                    "timestamp": time.time(),
                    "group_id":  gid,
                    "members":   members,
                    "cohesion":  round(cohesion, 3),
                    "reason":    alert_reason,
                })

            groups.append(GroupDetection(
                group_id     = gid,
                members      = members,
                cohesion     = round(cohesion, 3),
                formation    = formation,
                alert        = alert,
                alert_reason = alert_reason,
            ))

        return groups

    def get_all_links(self) -> list[SocialLink]:
        return list(self.links.values())

    def summary(self) -> dict:
        groups = self.detect_groups()
        return {
            "persons_tracked":  len(self.buffers),
            "active_links":     len(self.links),
            "groups_detected":  len(groups),
            "total_alerts":     len(self.alert_log),
            "groups":           [
                {
                    "id":        g.group_id,
                    "members":   g.members,
                    "cohesion":  g.cohesion,
                    "formation": g.formation,
                    "alert":     g.alert,
                }
                for g in groups
            ],
        }

    def reset_all(self):
        self.buffers.clear()
        self.links.clear()
        self.alert_log.clear()
        self.frame_count = 0


if __name__ == "__main__":
    print("=" * 60)
    print("Social Graph Intelligence — PhantomEye")
    print("=" * 60)

    engine = SocialGraphEngine(proximity_px=150)

    # Simulate 3 persons entering separately but moving together
    print("\n[SIM] Persons 1, 2 move together — Person 3 moves independently")

    for frame in range(80):
        t = frame * 0.5

        # Person 1 and 2 — synchronized movement (same velocity pattern)
        p1x = int(100 + t * 3 + np.random.randn() * 2)
        p1y = int(240 + np.sin(t) * 20 + np.random.randn() * 2)

        p2x = int(140 + t * 3 + np.random.randn() * 2)   # close to P1, same direction
        p2y = int(260 + np.sin(t) * 20 + np.random.randn() * 2)

        # Person 3 — independent movement
        p3x = int(400 + np.cos(t * 0.5) * 80 + np.random.randn() * 5)
        p3y = int(200 + np.sin(t * 0.7) * 60 + np.random.randn() * 5)

        engine.observe(1, (p1x, p1y))
        engine.observe(2, (p2x, p2y))
        engine.observe(3, (p3x, p3y))

        if frame % 10 == 0:
            engine._update_links()

    print("\n[LINKS DETECTED]")
    for link in engine.get_all_links():
        print(f"  Person {link.person_a} <-> Person {link.person_b}")
        print(f"    Strength: {link.strength:.3f} | Type: {link.link_type}")
        print(f"    Frames: {link.frame_count} | Evidence: {link.evidence}")

    print("\n[GROUPS DETECTED]")
    groups = engine.detect_groups()
    for g in groups:
        print(f"  Group {g.group_id}: {g.members} | Formation: {g.formation} | Cohesion: {g.cohesion:.3f}")
        if g.alert:
            print(f"  ALERT: {g.alert_reason}")

    print("\n[SESSION SUMMARY]")
    s = engine.summary()
    for k, v in s.items():
        if k != "groups":
            print(f"  {k}: {v}")
    print("=" * 60)
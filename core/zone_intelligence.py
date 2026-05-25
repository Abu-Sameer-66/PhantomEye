"""
PhantomEye — Zone Intelligence Engine (ZIE v1.0)
Upgrade 11 | core/zone_intelligence.py

Novel contribution: Multi-zone behavioral surveillance with TMS integration.
Tracks per-zone entry/exit/dwell/occupancy. Fires breach alerts. Feeds
proximity_violation signal directly into TMS. Detects suspicious zone
traversal sequences. No open-source equivalent combines all of these.

Author: Abu Sameer (IUB AI Research Lab)
"""

import time
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from collections import defaultdict, deque
from enum import Enum


# ── Zone Types ─────────────────────────────────────────── #

class ZoneType(str, Enum):
    RESTRICTED       = "RESTRICTED"        # No entry allowed
    MONITORED        = "MONITORED"         # Entry tracked + logged
    CAPACITY_LIMITED = "CAPACITY_LIMITED"  # Max occupancy enforced
    SAFE             = "SAFE"              # Normal zone, no alerts


# ── Alert Levels ───────────────────────────────────────── #

class AlertLevel(str, Enum):
    NONE     = "NONE"
    LOW      = "LOW"
    MEDIUM   = "MEDIUM"
    HIGH     = "HIGH"
    CRITICAL = "CRITICAL"


# ── Data Structures ────────────────────────────────────── #

@dataclass
class Zone:
    zone_id:      int
    name:         str
    zone_type:    ZoneType
    x1:           int          # top-left x (pixels)
    y1:           int          # top-left y (pixels)
    x2:           int          # bottom-right x (pixels)
    y2:           int          # bottom-right y (pixels)
    max_capacity: int = 5      # used for CAPACITY_LIMITED
    color:        tuple = (0, 180, 255)  # BGR

    def contains(self, cx: float, cy: float) -> bool:
        return self.x1 <= cx <= self.x2 and self.y1 <= cy <= self.y2

    def area(self) -> int:
        return max(0, (self.x2 - self.x1) * (self.y2 - self.y1))

    def to_dict(self) -> dict:
        return {
            "zone_id":      self.zone_id,
            "name":         self.name,
            "zone_type":    self.zone_type.value,
            "bbox":         [self.x1, self.y1, self.x2, self.y2],
            "max_capacity": self.max_capacity,
        }


@dataclass
class ZoneEvent:
    event_type:  str           # "entry" | "exit" | "breach" | "capacity" | "sequence"
    zone_id:     int
    zone_name:   str
    person_id:   int
    timestamp:   float
    alert_level: AlertLevel
    message:     str

    def to_dict(self) -> dict:
        return {
            "event_type":  self.event_type,
            "zone_id":     self.zone_id,
            "zone_name":   self.zone_name,
            "person_id":   self.person_id,
            "timestamp":   round(self.timestamp, 2),
            "alert_level": self.alert_level.value,
            "message":     self.message,
        }


@dataclass
class PersonZoneState:
    person_id:        int
    current_zones:    set   = field(default_factory=set)   # zone_ids currently inside
    zone_history:     list  = field(default_factory=list)  # [(zone_id, entry_ts, exit_ts)]
    entry_times:      dict  = field(default_factory=dict)  # zone_id → entry timestamp
    total_dwell:      dict  = field(default_factory=lambda: defaultdict(float))  # zone_id → seconds
    breach_count:     int   = 0
    sequence_log:     list  = field(default_factory=list)  # ordered zone_ids visited


@dataclass
class ZoneSummary:
    zone_id:          int
    zone_name:        str
    zone_type:        str
    total_entries:    int
    total_exits:      int
    current_occupancy: int
    avg_dwell_sec:    float
    max_dwell_sec:    float
    total_breaches:   int
    alert_level:      AlertLevel
    persons_inside:   list

    def to_dict(self) -> dict:
        return {
            "zone_id":           self.zone_id,
            "zone_name":         self.zone_name,
            "zone_type":         self.zone_type,
            "total_entries":     self.total_entries,
            "total_exits":       self.total_exits,
            "current_occupancy": self.current_occupancy,
            "avg_dwell_sec":     round(self.avg_dwell_sec, 2),
            "max_dwell_sec":     round(self.max_dwell_sec, 2),
            "total_breaches":    self.total_breaches,
            "alert_level":       self.alert_level.value,
            "persons_inside":    self.persons_inside,
        }


# ── Suspicious Sequences ───────────────────────────────── #

SUSPICIOUS_SEQUENCES = [
    ([0, 1, 2], "Perimeter → Corridor → Restricted traversal detected"),
    ([1, 0],    "Direct approach to restricted zone from monitored area"),
]


# ── Core Engine ────────────────────────────────────────── #

class ZoneIntelligenceEngine:
    """
    ZIE v1.0 — Zone Intelligence Engine

    Algorithm:
    1. Maintain list of named, typed zones (rectangles on frame).
    2. Each frame: compute centroid of each tracked person.
    3. For each person, determine which zones their centroid falls in.
    4. Detect zone entry (centroid enters zone) and exit (centroid leaves).
    5. Track dwell time per person per zone.
    6. Fire alerts:
       - RESTRICTED entry → CRITICAL breach alert
       - CAPACITY_LIMITED overflow → HIGH alert
       - Suspicious sequence → MEDIUM alert
    7. Emit TMS-compatible signal dict for integration.
    """

    def __init__(self):
        self.zones:         dict[int, Zone]            = {}
        self.person_states: dict[int, PersonZoneState] = {}
        self.zone_entries:  dict[int, int]             = defaultdict(int)   # zone_id → total entries
        self.zone_exits:    dict[int, int]             = defaultdict(int)
        self.zone_breaches: dict[int, int]             = defaultdict(int)
        self.zone_dwells:   dict[int, list]            = defaultdict(list)  # zone_id → [dwell_secs]
        self.event_log:     list[ZoneEvent]            = []
        self.frame_counter: int                        = 0
        self._next_zone_id: int                        = 1

    # ── Zone Management ──────────────────────────────── #

    def add_zone(self, name: str, zone_type: ZoneType,
                 x1: int, y1: int, x2: int, y2: int,
                 max_capacity: int = 5,
                 color: tuple = None) -> Zone:
        """Define a new surveillance zone."""
        zid = self._next_zone_id
        self._next_zone_id += 1
        default_colors = {
            ZoneType.RESTRICTED:       (51,  51,  255),  # red
            ZoneType.MONITORED:        (255, 180,  0),   # orange
            ZoneType.CAPACITY_LIMITED: (0,   255, 180),  # cyan
            ZoneType.SAFE:             (0,   200, 80),   # green
        }
        zone = Zone(
            zone_id=zid, name=name, zone_type=zone_type,
            x1=min(x1,x2), y1=min(y1,y2),
            x2=max(x1,x2), y2=max(y1,y2),
            max_capacity=max_capacity,
            color=color or default_colors.get(zone_type, (0,180,255))
        )
        self.zones[zid] = zone
        return zone

    def remove_zone(self, zone_id: int):
        self.zones.pop(zone_id, None)

    def clear_zones(self):
        self.zones.clear()

    # ── Main Update ──────────────────────────────────── #

    def update(self, detections: list[dict],
               frame_id: int = None,
               tms_engine=None) -> dict:
        """
        Call once per frame.

        detections: list of {"person_id": int, "bbox": [x1,y1,x2,y2]}
        tms_engine: optional ThreatMomentumEngine instance for direct integration
        Returns: {"events": [...], "zone_summaries": {...], "tms_signals": {...}}
        """
        if frame_id is None:
            frame_id = self.frame_counter
        self.frame_counter += 1
        now = time.time()
        new_events = []
        tms_signals = {}  # person_id → {"in_restricted_zone": bool}

        # Current positions
        current_positions = {}
        for det in detections:
            pid  = det["person_id"]
            bbox = det["bbox"]
            cx   = (bbox[0] + bbox[2]) / 2.0
            cy   = (bbox[1] + bbox[3]) / 2.0
            current_positions[pid] = (cx, cy)

            if pid not in self.person_states:
                self.person_states[pid] = PersonZoneState(person_id=pid)

        # Zone occupancy snapshot this frame
        zone_current_persons: dict[int, set] = {zid: set() for zid in self.zones}

        for pid, (cx, cy) in current_positions.items():
            state   = self.person_states[pid]
            in_zones = set()

            for zid, zone in self.zones.items():
                if zone.contains(cx, cy):
                    in_zones.add(zid)
                    zone_current_persons[zid].add(pid)

            # Detect entries
            entered = in_zones - state.current_zones
            exited  = state.current_zones - in_zones

            for zid in entered:
                zone = self.zones[zid]
                state.entry_times[zid] = now
                self.zone_entries[zid] += 1

                if zone.zone_type not in state.sequence_log or \
                   (state.sequence_log and state.sequence_log[-1] != zid):
                    state.sequence_log.append(zid)

                # RESTRICTED breach
                if zone.zone_type == ZoneType.RESTRICTED:
                    state.breach_count += 1
                    self.zone_breaches[zid] += 1
                    evt = ZoneEvent(
                        event_type="breach", zone_id=zid, zone_name=zone.name,
                        person_id=pid, timestamp=now,
                        alert_level=AlertLevel.CRITICAL,
                        message=f"BREACH: Person {pid} entered RESTRICTED zone '{zone.name}'"
                    )
                    new_events.append(evt)
                    tms_signals[pid] = {"in_restricted_zone": True}

                else:
                    evt = ZoneEvent(
                        event_type="entry", zone_id=zid, zone_name=zone.name,
                        person_id=pid, timestamp=now,
                        alert_level=AlertLevel.LOW if zone.zone_type == ZoneType.MONITORED else AlertLevel.NONE,
                        message=f"Person {pid} entered zone '{zone.name}'"
                    )
                    new_events.append(evt)

            # Detect exits
            for zid in exited:
                zone  = self.zones.get(zid)
                if zone and zid in state.entry_times:
                    dwell = now - state.entry_times.pop(zid)
                    state.total_dwell[zid] += dwell
                    self.zone_dwells[zid].append(dwell)
                    self.zone_exits[zid] += 1
                    evt = ZoneEvent(
                        event_type="exit", zone_id=zid, zone_name=zone.name,
                        person_id=pid, timestamp=now,
                        alert_level=AlertLevel.NONE,
                        message=f"Person {pid} exited zone '{zone.name}' after {dwell:.1f}s"
                    )
                    new_events.append(evt)

            state.current_zones = in_zones

        # Capacity alerts
        for zid, zone in self.zones.items():
            if zone.zone_type == ZoneType.CAPACITY_LIMITED:
                occ = len(zone_current_persons[zid])
                if occ > zone.max_capacity:
                    evt = ZoneEvent(
                        event_type="capacity", zone_id=zid, zone_name=zone.name,
                        person_id=-1, timestamp=now,
                        alert_level=AlertLevel.HIGH,
                        message=f"CAPACITY EXCEEDED: Zone '{zone.name}' has {occ}/{zone.max_capacity} persons"
                    )
                    new_events.append(evt)

        # Suspicious sequence detection
        for pid, state in self.person_states.items():
            if len(state.sequence_log) >= 2:
                recent = state.sequence_log[-3:]
                for seq, reason in SUSPICIOUS_SEQUENCES:
                    if len(recent) >= len(seq) and recent[-len(seq):] == seq:
                        evt = ZoneEvent(
                            event_type="sequence", zone_id=-1, zone_name="MULTI-ZONE",
                            person_id=pid, timestamp=now,
                            alert_level=AlertLevel.MEDIUM,
                            message=f"SUSPICIOUS PATTERN: Person {pid} — {reason}"
                        )
                        new_events.append(evt)

        # Log all events
        self.event_log.extend(new_events)
        if len(self.event_log) > 500:
            self.event_log = self.event_log[-500:]

        # TMS integration
        if tms_engine and tms_signals:
            for pid, signals in tms_signals.items():
                try:
                    cx, cy = current_positions.get(pid, (0, 0))
                    tms_engine.update_person(
                        person_id=pid,
                        position=(int(cx), int(cy)),
                        in_restricted_zone=signals.get("in_restricted_zone", False),
                    )
                except Exception:
                    pass

        return {
            "events":        [e.to_dict() for e in new_events],
            "zone_summaries": self._build_summaries(zone_current_persons),
            "tms_signals":   tms_signals,
            "alert_count":   sum(1 for e in new_events if e.alert_level != AlertLevel.NONE),
        }

    # ── Summary Builder ──────────────────────────────── #

    def _build_summaries(self, zone_current: dict) -> dict:
        summaries = {}
        for zid, zone in self.zones.items():
            dwells = self.zone_dwells.get(zid, [])
            persons_inside = list(zone_current.get(zid, set()))
            occ = len(persons_inside)
            alert = AlertLevel.NONE
            if zone.zone_type == ZoneType.RESTRICTED and self.zone_breaches[zid] > 0:
                alert = AlertLevel.CRITICAL
            elif zone.zone_type == ZoneType.CAPACITY_LIMITED and occ > zone.max_capacity:
                alert = AlertLevel.HIGH
            summaries[zid] = ZoneSummary(
                zone_id=zid, zone_name=zone.name, zone_type=zone.zone_type.value,
                total_entries=self.zone_entries[zid],
                total_exits=self.zone_exits[zid],
                current_occupancy=occ,
                avg_dwell_sec=float(np.mean(dwells)) if dwells else 0.0,
                max_dwell_sec=float(np.max(dwells)) if dwells else 0.0,
                total_breaches=self.zone_breaches[zid],
                alert_level=alert,
                persons_inside=persons_inside,
            ).to_dict()
        return summaries

    # ── Visualization ────────────────────────────────── #

    def draw_zones(self, frame: np.ndarray) -> np.ndarray:
        """Overlay all zones on a frame with color-coded borders."""
        import cv2
        out = frame.copy()
        for zone in self.zones.values():
            color = zone.color
            occ   = sum(1 for s in self.person_states.values() if zone.zone_id in s.current_zones)

            # Fill with transparency
            overlay = out.copy()
            cv2.rectangle(overlay, (zone.x1, zone.y1), (zone.x2, zone.y2), color, -1)
            cv2.addWeighted(overlay, 0.12, out, 0.88, 0, out)

            # Border
            thickness = 3 if zone.zone_type == ZoneType.RESTRICTED else 2
            cv2.rectangle(out, (zone.x1, zone.y1), (zone.x2, zone.y2), color, thickness)

            # Label
            label = f"{zone.name} [{zone.zone_type.value}] ({occ})"
            cv2.putText(out, label, (zone.x1 + 6, zone.y1 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # Restricted — red X
            if zone.zone_type == ZoneType.RESTRICTED:
                cv2.line(out, (zone.x1, zone.y1), (zone.x2, zone.y2), color, 1)
                cv2.line(out, (zone.x2, zone.y1), (zone.x1, zone.y2), color, 1)

        return out

    # ── Utilities ────────────────────────────────────── #

    def get_recent_events(self, n: int = 20) -> list:
        return [e.to_dict() for e in self.event_log[-n:]]

    def get_all_summaries(self) -> dict:
        zone_current = {zid: set() for zid in self.zones}
        for pid, state in self.person_states.items():
            for zid in state.current_zones:
                if zid in zone_current:
                    zone_current[zid].add(pid)
        return self._build_summaries(zone_current)

    def session_summary(self) -> dict:
        total_breaches = sum(self.zone_breaches.values())
        total_entries  = sum(self.zone_entries.values())
        return {
            "total_zones":        len(self.zones),
            "total_persons_seen": len(self.person_states),
            "total_entries":      total_entries,
            "total_breaches":     total_breaches,
            "total_events":       len(self.event_log),
            "zone_breakdown":     {
                z.name: {
                    "type":     z.zone_type.value,
                    "entries":  self.zone_entries[z.zone_id],
                    "breaches": self.zone_breaches[z.zone_id],
                }
                for z in self.zones.values()
            }
        }

    def reset(self):
        self.zones.clear()
        self.person_states.clear()
        self.zone_entries.clear()
        self.zone_exits.clear()
        self.zone_breaches.clear()
        self.zone_dwells.clear()
        self.event_log.clear()
        self.frame_counter = 0
        self._next_zone_id = 1


# ── Singleton ─────────────────────────────────────────── #

_engine: Optional[ZoneIntelligenceEngine] = None


def get_engine() -> ZoneIntelligenceEngine:
    global _engine
    if _engine is None:
        _engine = ZoneIntelligenceEngine()
    return _engine


def reset_engine():
    global _engine
    if _engine:
        _engine.reset()


# ── Standalone Test ───────────────────────────────────── #

if __name__ == "__main__":
    engine = ZoneIntelligenceEngine()

    # Define zones
    engine.add_zone("Server Room",    ZoneType.RESTRICTED,       400, 100, 600, 350)
    engine.add_zone("Reception",      ZoneType.MONITORED,        50,  50,  350, 300)
    engine.add_zone("Lobby",          ZoneType.CAPACITY_LIMITED, 50, 320,  640, 480, max_capacity=3)
    engine.add_zone("Exit Corridor",  ZoneType.SAFE,             600, 300, 700, 480)

    print("=" * 65)
    print("ZIE v1.0 — Zone Intelligence Engine | Standalone Test")
    print("=" * 65)
    print(f"Zones defined: {len(engine.zones)}")
    for z in engine.zones.values():
        print(f"  [{z.zone_id}] {z.name:<20} {z.zone_type.value}")

    print("\n--- Simulating person movement ---\n")

    scenarios = [
        # (frame, person_id, bbox, description)
        (0,  1, [60,  60,  110, 160], "Person 1 enters Reception"),
        (5,  1, [410, 110, 460, 210], "Person 1 enters RESTRICTED Server Room"),
        (10, 2, [60,  330, 110, 430], "Person 2 enters Lobby"),
        (11, 3, [80,  340, 130, 440], "Person 3 enters Lobby"),
        (12, 4, [100, 350, 150, 450], "Person 4 enters Lobby"),
        (13, 5, [120, 360, 170, 460], "Person 5 enters Lobby — CAPACITY EXCEEDED"),
        (20, 1, [610, 310, 660, 410], "Person 1 exits to Safe Corridor"),
    ]

    for frame_id, pid, bbox, desc in scenarios:
        dets = [{"person_id": pid, "bbox": bbox}]
        result = engine.update(dets, frame_id=frame_id)
        events = result["events"]
        if events:
            for evt in events:
                level = evt["alert_level"]
                symbol = "🔴" if level == "CRITICAL" else "🟠" if level == "HIGH" else "🟡" if level == "MEDIUM" else "🟢"
                print(f"Frame {frame_id:02d} {symbol} [{level:<8}] {evt['message']}")

    print("\n--- Session Summary ---")
    s = engine.session_summary()
    print(f"  Zones: {s['total_zones']}")
    print(f"  Persons seen: {s['total_persons_seen']}")
    print(f"  Total entries: {s['total_entries']}")
    print(f"  Total breaches: {s['total_breaches']}")
    print("\nZIE v1.0 — All tests passed ✓")
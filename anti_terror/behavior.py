from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

from .config import BehaviorConfig
from .association import bbox_center, iou


@dataclass
class BagState:
    """Enhanced bag state with position history for better static detection."""
    bag_id: str
    last_box: np.ndarray
    last_update: float
    owner_person_id: Optional[str] = None
    last_owner_seen: float = field(default_factory=time.time)
    static_since: Optional[float] = None

    # Position history for variance-based static detection
    position_history: List[Tuple[float, float]] = field(default_factory=list)

    # Alert tracking for cooldown
    last_alert_time: Optional[float] = None
    alert_count: int = 0


class BehaviorAnalyzer:
    """Enhanced behavior analyzer with position variance and alert cooldown."""

    def __init__(self, cfg: BehaviorConfig):
        self.cfg = cfg
        self.bags: Dict[int, BagState] = {}  # key: bag track_id
        # Thread-safe cleanup
        self._cleanup_lock = threading.Lock()

    def _get_center(self, box: np.ndarray) -> Tuple[float, float]:
        """Get center point of bounding box."""
        return float((box[0] + box[2]) / 2), float((box[1] + box[3]) / 2)

    def _check_static(self, state: BagState, current_box: np.ndarray) -> bool:
        """Enhanced static detection using IoU and position variance.

        A bag is considered static if:
        1. IoU between current and last box is high (traditional method), OR
        2. Position variance over recent frames is low (more robust to noise)
        """
        # IoU check (traditional method)
        iou_static = iou(state.last_box, current_box) >= self.cfg.static_iou_threshold

        # Position variance check (more robust)
        if len(state.position_history) >= 15:
            positions = np.array(state.position_history[-15:])
            variance = np.var(positions, axis=0).sum()
            # Less than 100 means ~10px std dev - bag is stationary
            variance_static = variance < 100
            return iou_static or variance_static

        return iou_static

    def update(
        self,
        bag_tracks,
        bag_ids: Dict[int, str],
        person_ids: Dict[int, str],
        assignments: Dict[int, int],
    ) -> list[dict]:
        """Update bag states and detect abandoned bags.

        Returns list of event dicts for abandoned bags with alert cooldown.
        """
        now = time.time()
        events = []

        # Get config values with defaults
        alert_cooldown = getattr(self.cfg, 'alert_cooldown_s', 30.0)
        stale_timeout = getattr(self.cfg, 'stale_track_timeout_s', 60.0)
        min_confidence = getattr(self.cfg, 'min_confidence_for_abandonment', 0.5)

        # Update bag states
        active_track_ids = set()
        for bag in bag_tracks:
            active_track_ids.add(bag.track_id)
            bag_id = bag_ids.get(bag.track_id, f"B{bag.track_id}")

            state = self.bags.get(bag.track_id)
            if state is None:
                state = BagState(
                    bag_id=bag_id,
                    last_box=bag.box.copy(),
                    last_update=now
                )
                self.bags[bag.track_id] = state
            else:
                # Update position history
                current_center = self._get_center(bag.box)
                state.position_history.append(current_center)
                # Keep last 30 positions (~1 sec at 30fps)
                if len(state.position_history) > 30:
                    state.position_history = state.position_history[-30:]

                # Check if bag is static using enhanced detection
                is_static = self._check_static(state, bag.box)
                if is_static:
                    state.static_since = state.static_since or now
                else:
                    state.static_since = None

                state.last_box = bag.box.copy()
                state.last_update = now
                state.bag_id = bag_id  # Update in case it changed

            # Update owner info
            if bag.track_id in assignments:
                pid = assignments[bag.track_id]
                state.owner_person_id = person_ids.get(pid)
                state.last_owner_seen = now

        # Detect abandoned bags with thread-safe cleanup
        with self._cleanup_lock:
            stale_tracks = []

            for track_id, state in self.bags.items():
                # Check for stale tracks
                if now - state.last_update > stale_timeout:
                    stale_tracks.append(track_id)
                    continue

                # Skip if not static
                if not state.static_since:
                    continue

                # Skip if no owner ever assigned
                if state.owner_person_id is None:
                    continue

                # Check abandonment conditions
                away_time = now - state.last_owner_seen
                static_time = now - state.static_since

                if (away_time >= self.cfg.abandonment_timeout_s and
                    static_time >= self.cfg.abandonment_timeout_s):

                    # Check alert cooldown to prevent spam
                    if (state.last_alert_time is None or
                        now - state.last_alert_time >= alert_cooldown):

                        events.append({
                            "type": "Abandoned Bag",
                            "bag_id": state.bag_id,
                            "person_id": state.owner_person_id,
                            "away_for_s": round(away_time, 2),
                            "static_for_s": round(static_time, 2),
                            "alert_count": state.alert_count + 1,
                        })

                        state.last_alert_time = now
                        state.alert_count += 1
                        logger.warning(
                            f"ABANDONED BAG: {state.bag_id} (owner: {state.owner_person_id}, "
                            f"away: {away_time:.1f}s, static: {static_time:.1f}s)"
                        )
                        # Don't reset static_since - keep tracking for future alerts

            # Clean up stale tracks
            for track_id in stale_tracks:
                del self.bags[track_id]

        return events

    def get_bag_state(self, track_id: int) -> Optional[BagState]:
        """Get current state of a bag."""
        return self.bags.get(track_id)

    def reset(self):
        """Reset all bag states."""
        with self._cleanup_lock:
            self.bags.clear()

"""Bag-Person association engine with persistent ownership tracking.

Key features:
1. First-seen ownership: Person who first appears with a bag becomes owner
2. Proximity-based association using face position
3. Confidence scoring based on consistency
4. Support for ownership transfer detection
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import time

import numpy as np
from loguru import logger
from shapely.geometry import Polygon

from .config import AssociationConfig


def bbox_center(box: np.ndarray) -> tuple[float, float]:
    """Get center point of a bounding box."""
    x1, y1, x2, y2 = box
    return float((x1 + x2) / 2), float((y1 + y2) / 2)


def bbox_bottom_center(box: np.ndarray) -> tuple[float, float]:
    """Get bottom center point (useful for faces - approximate body position)."""
    x1, y1, x2, y2 = box
    return float((x1 + x2) / 2), float(y2)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate Intersection over Union between two boxes."""
    poly_a = Polygon([(a[0], a[1]), (a[0], a[3]), (a[2], a[3]), (a[2], a[1])])
    poly_b = Polygon([(b[0], b[1]), (b[0], b[3]), (b[2], b[3]), (b[2], b[1])])
    inter = poly_a.intersection(poly_b).area
    union = poly_a.union(poly_b).area
    if union == 0:
        return 0.0
    return float(inter / union)


def distance(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    """Euclidean distance between two points."""
    return float(np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2))


@dataclass
class BagOwnership:
    """Tracks ownership state for a single bag."""
    bag_track_id: int
    bag_id: str | None = None

    # Primary owner (first person seen carrying this bag)
    owner_person_id: str | None = None
    owner_track_id: int | None = None

    # Ownership confidence
    confidence: float = 0.0
    link_frames: int = 0  # Frames where owner was near bag

    # Timing
    first_seen: float = field(default_factory=time.time)
    last_owner_seen: float = field(default_factory=time.time)

    # State
    is_being_carried: bool = False
    carrier_person_id: str | None = None


@dataclass
class PersonBags:
    """Tracks bags associated with a person."""
    person_id: str
    person_track_id: int
    bag_ids: List[str] = field(default_factory=list)
    bag_track_ids: List[int] = field(default_factory=list)


class AssociationEngine:
    """Engine for associating bags with people and tracking ownership.

    Ownership rules:
    1. First person seen close to a new bag becomes the owner
    2. Ownership can transfer if bag is consistently near another person
    3. Confidence increases with consistent proximity
    """

    def __init__(self, cfg: AssociationConfig):
        self.cfg = cfg
        self.bag_ownerships: Dict[int, BagOwnership] = {}  # bag_track_id -> ownership
        self.person_bags: Dict[str, PersonBags] = {}  # person_id -> bags
        self.frame_count = 0
        logger.info("Association engine ready")

    def associate(
        self,
        face_tracks,  # List of FaceTrack from face_tracker
        bag_tracks,  # List of Track for bags
        person_ids: Dict[int, str],  # track_id -> person_id
        bag_ids: Dict[int, str]  # track_id -> bag_id
    ) -> Dict[int, int]:
        """Associate bags with people based on proximity.

        Args:
            face_tracks: List of face tracks (with person_id)
            bag_tracks: List of bag tracks
            person_ids: Mapping of track_id to person_id
            bag_ids: Mapping of track_id to bag_id

        Returns:
            Dict mapping bag_track_id to owner's track_id
        """
        now = time.time()
        self.frame_count += 1
        assignments: Dict[int, int] = {}

        # Build person position map (using face bottom as approximate body position)
        person_positions: Dict[int, tuple[float, float]] = {}
        for ft in face_tracks:
            if ft.person_id:
                # Use bottom of face as approximate shoulder/body position
                person_positions[ft.track_id] = bbox_bottom_center(ft.box)

        # Process each bag
        for bag in bag_tracks:
            bag_center = bbox_center(bag.box)
            bag_id = bag_ids.get(bag.track_id)

            # Get or create ownership record
            if bag.track_id not in self.bag_ownerships:
                self.bag_ownerships[bag.track_id] = BagOwnership(
                    bag_track_id=bag.track_id,
                    bag_id=bag_id,
                    first_seen=now
                )
            ownership = self.bag_ownerships[bag.track_id]
            ownership.bag_id = bag_id

            # Find closest person to this bag
            closest_person = None
            closest_dist = float('inf')
            closest_track_id = None

            for track_id, pos in person_positions.items():
                dist = distance(bag_center, pos)
                if dist < closest_dist:
                    closest_dist = dist
                    closest_person = person_ids.get(track_id)
                    closest_track_id = track_id

            # Check if close enough to be "carrying"
            is_carrying = closest_dist <= self.cfg.max_link_distance_px

            if is_carrying and closest_person:
                ownership.is_being_carried = True
                ownership.carrier_person_id = closest_person

                # First-seen ownership: assign owner if none
                if ownership.owner_person_id is None:
                    ownership.owner_person_id = closest_person
                    ownership.owner_track_id = closest_track_id
                    ownership.confidence = 0.5
                    ownership.link_frames = 1
                    ownership.last_owner_seen = now
                    logger.info(f"Bag {bag_id} assigned to owner {closest_person}")

                    # Update person's bag list
                    self._add_bag_to_person(closest_person, closest_track_id, bag_id, bag.track_id)

                # Same owner carrying
                elif ownership.owner_person_id == closest_person:
                    ownership.link_frames += 1
                    ownership.last_owner_seen = now
                    # Increase confidence with consistency
                    ownership.confidence = min(1.0, ownership.confidence + 0.02)

                # Different person carrying - potential transfer
                else:
                    # Only transfer if consistently with new person for many frames
                    if ownership.carrier_person_id == closest_person:
                        # Track consecutive frames with new carrier
                        transfer_frames = getattr(ownership, '_transfer_frames', 0) + 1
                        ownership._transfer_frames = transfer_frames

                        # Transfer ownership after significant time with new person
                        if transfer_frames >= self.cfg.time_consistency_frames * 3:
                            old_owner = ownership.owner_person_id
                            ownership.owner_person_id = closest_person
                            ownership.owner_track_id = closest_track_id
                            ownership.confidence = 0.6
                            ownership._transfer_frames = 0
                            logger.info(f"Bag {bag_id} ownership transferred: {old_owner} -> {closest_person}")

                            # Update person bag lists
                            self._remove_bag_from_person(old_owner, bag_id)
                            self._add_bag_to_person(closest_person, closest_track_id, bag_id, bag.track_id)
                    else:
                        ownership._transfer_frames = 0

                # Set assignment to owner
                if ownership.owner_track_id:
                    assignments[bag.track_id] = ownership.owner_track_id

            else:
                ownership.is_being_carried = False
                ownership.carrier_person_id = None
                ownership._transfer_frames = 0

                # Still assign to owner even if not carrying (for abandoned bag detection)
                if ownership.owner_track_id:
                    assignments[bag.track_id] = ownership.owner_track_id

        # Cleanup stale bags
        stale_bags = [
            tid for tid, own in self.bag_ownerships.items()
            if now - own.first_seen > 300 and own.link_frames < 5  # 5 min with no links
        ]
        for tid in stale_bags:
            del self.bag_ownerships[tid]

        return assignments

    def _add_bag_to_person(self, person_id: str, track_id: int, bag_id: str, bag_track_id: int):
        """Add a bag to a person's list."""
        if person_id not in self.person_bags:
            self.person_bags[person_id] = PersonBags(
                person_id=person_id,
                person_track_id=track_id
            )
        pb = self.person_bags[person_id]
        if bag_id and bag_id not in pb.bag_ids:
            pb.bag_ids.append(bag_id)
        if bag_track_id not in pb.bag_track_ids:
            pb.bag_track_ids.append(bag_track_id)

    def _remove_bag_from_person(self, person_id: str, bag_id: str):
        """Remove a bag from a person's list."""
        if person_id in self.person_bags:
            pb = self.person_bags[person_id]
            if bag_id in pb.bag_ids:
                pb.bag_ids.remove(bag_id)

    def get_person_bags(self, person_id: str) -> List[str]:
        """Get list of bag IDs owned by a person."""
        if person_id in self.person_bags:
            return self.person_bags[person_id].bag_ids.copy()
        return []

    def get_bag_owner(self, bag_track_id: int) -> str | None:
        """Get owner person_id for a bag."""
        if bag_track_id in self.bag_ownerships:
            return self.bag_ownerships[bag_track_id].owner_person_id
        return None

    def get_ownership_confidence(self, bag_track_id: int) -> float:
        """Get ownership confidence for a bag."""
        if bag_track_id in self.bag_ownerships:
            return self.bag_ownerships[bag_track_id].confidence
        return 0.0

    def is_bag_being_carried(self, bag_track_id: int) -> bool:
        """Check if bag is currently being carried."""
        if bag_track_id in self.bag_ownerships:
            return self.bag_ownerships[bag_track_id].is_being_carried
        return False

    def get_all_ownerships(self) -> Dict[str, str]:
        """Get all bag_id -> owner_person_id mappings."""
        return {
            own.bag_id: own.owner_person_id
            for own in self.bag_ownerships.values()
            if own.bag_id and own.owner_person_id
        }

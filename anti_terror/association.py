"""Bag-Person association engine with persistent ownership tracking.

Key features:
1. Person-ID based ownership (survives track_id changes after occlusion)
2. Temporal voting for crowd disambiguation
3. Confidence decay when owner not visible
4. Multi-candidate history for stable ownership
5. Support for ownership transfer detection
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Deque
from collections import deque
import time
import math

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


def estimate_body_from_face(face_box: np.ndarray) -> np.ndarray:
    """Estimate body bounding box from face position.

    Assumes: body is roughly 2.5x face width and 4.5x face height,
    positioned below the face.
    """
    x1, y1, x2, y2 = face_box
    face_width = x2 - x1
    face_height = y2 - y1

    # Body width ~2.5x face width, height ~4.5x face height
    body_width = face_width * 2.5
    body_height = face_height * 4.5

    center_x = (x1 + x2) / 2
    body_x1 = center_x - body_width / 2
    body_x2 = center_x + body_width / 2
    body_y1 = y2  # Start from chin
    body_y2 = y2 + body_height

    return np.array([body_x1, body_y1, body_x2, body_y2])


@dataclass
class CandidateObservation:
    """Single observation of a candidate for bag ownership."""
    person_id: str
    track_id: int
    timestamp: float
    distance: float
    iou: float
    is_carrying: bool


@dataclass
class CandidateHistory:
    """History of observations for a candidate owner."""
    person_id: str
    observations: Deque[CandidateObservation] = field(default_factory=lambda: deque(maxlen=60))
    total_frames: int = 0
    carrying_frames: int = 0
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)

    def add_observation(self, obs: CandidateObservation) -> None:
        """Add a new observation."""
        self.observations.append(obs)
        self.total_frames += 1
        if obs.is_carrying:
            self.carrying_frames += 1
        self.last_seen = obs.timestamp

    def get_temporal_score(self, current_time: float, window_s: float = 2.0) -> float:
        """Calculate temporal consistency score (0-1).

        Higher score = more consistent presence near bag.
        """
        if not self.observations:
            return 0.0

        # Count recent observations
        cutoff = current_time - window_s
        recent = [o for o in self.observations if o.timestamp >= cutoff]

        if not recent:
            return 0.0

        # Ratio of frames present
        expected_frames = window_s * 30  # ~30fps
        presence_ratio = min(1.0, len(recent) / expected_frames)

        # Carrying bonus
        carrying_ratio = sum(1 for o in recent if o.is_carrying) / len(recent)

        return 0.6 * presence_ratio + 0.4 * carrying_ratio

    def get_avg_distance(self, window_s: float = 2.0) -> float:
        """Get average distance in recent window."""
        current_time = time.time()
        cutoff = current_time - window_s
        recent = [o for o in self.observations if o.timestamp >= cutoff]

        if not recent:
            return float('inf')

        return sum(o.distance for o in recent) / len(recent)

    def get_avg_iou(self, window_s: float = 2.0) -> float:
        """Get average IoU in recent window."""
        current_time = time.time()
        cutoff = current_time - window_s
        recent = [o for o in self.observations if o.timestamp >= cutoff]

        if not recent:
            return 0.0

        return sum(o.iou for o in recent) / len(recent)


@dataclass
class BagOwnership:
    """Tracks ownership state for a single bag.

    Uses person_id as primary key (survives track_id changes).
    Maintains candidate histories for temporal voting.
    """
    bag_track_id: int
    bag_id: str | None = None

    # Primary owner (person_id is the stable identifier)
    owner_person_id: str | None = None
    owner_track_id: int | None = None  # Current track_id (may change)

    # Ownership confidence and validation
    confidence: float = 0.0
    base_confidence: float = 0.0  # Confidence before decay starts
    link_frames: int = 0  # Frames where owner was near bag
    validated: bool = False  # True after minimum_validation_frames
    validation_frames: int = 0

    # Timing
    first_seen: float = field(default_factory=time.time)
    last_owner_seen: float = field(default_factory=time.time)
    last_any_person_nearby: float = field(default_factory=time.time)
    decay_start_time: float | None = None  # When confidence decay started

    # State
    is_being_carried: bool = False
    carrier_person_id: str | None = None

    # Candidate histories for temporal voting (person_id -> history)
    candidate_histories: Dict[str, CandidateHistory] = field(default_factory=dict)

    # Ambiguity tracking
    is_ambiguous: bool = False
    ambiguous_candidates: List[str] = field(default_factory=list)

    # Transfer tracking
    transfer_candidate_id: str | None = None
    transfer_frames: int = 0

    def get_or_create_history(self, person_id: str) -> CandidateHistory:
        """Get or create candidate history for a person."""
        if person_id not in self.candidate_histories:
            self.candidate_histories[person_id] = CandidateHistory(person_id=person_id)
        return self.candidate_histories[person_id]


@dataclass
class PersonBags:
    """Tracks bags associated with a person."""
    person_id: str
    person_track_id: int
    bag_ids: List[str] = field(default_factory=list)
    bag_track_ids: List[int] = field(default_factory=list)


class AssociationEngine:
    """Engine for associating bags with people and tracking ownership.

    Features:
    1. Person-ID based ownership (survives track_id changes)
    2. Temporal voting for crowd disambiguation
    3. Confidence decay when owner not visible
    4. Ambiguity detection and handling
    """

    def __init__(self, cfg: AssociationConfig):
        self.cfg = cfg
        self.bag_ownerships: Dict[int, BagOwnership] = {}  # bag_track_id -> ownership
        self.person_bags: Dict[str, PersonBags] = {}  # person_id -> bags
        self.frame_count = 0

        # Track ID to person ID mapping (updated by face tracker)
        self.track_to_person: Dict[int, str] = {}

        logger.info("Association engine ready with temporal voting")

    def associate(
        self,
        face_tracks,  # List of FaceTrack from face_tracker
        bag_tracks,  # List of Track for bags
        person_ids: Dict[int, str],  # track_id -> person_id
        bag_ids: Dict[int, str]  # track_id -> bag_id
    ) -> Dict[int, int]:
        """Associate bags with people using temporal voting.

        Uses person_id as the primary key (survives track_id changes).
        Implements temporal voting for crowd disambiguation.
        Applies confidence decay when owner is not visible.

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

        # Update track_to_person mapping
        self.track_to_person = person_ids.copy()

        # Build person info with body estimation (indexed by person_id)
        person_info: Dict[str, dict] = {}
        person_to_track: Dict[str, int] = {}

        for ft in face_tracks:
            if ft.person_id:
                body_box = estimate_body_from_face(ft.box)
                person_info[ft.person_id] = {
                    'track_id': ft.track_id,
                    'person_id': ft.person_id,
                    'face_center': bbox_center(ft.box),
                    'body_center': bbox_center(body_box),
                    'body_box': body_box,
                }
                person_to_track[ft.person_id] = ft.track_id

        # Set of all visible person_ids
        visible_persons = set(person_info.keys())

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

            # Collect ALL candidates within far_distance
            all_candidates = []
            carrying_dist = getattr(self.cfg, 'carrying_distance_px', 80)
            carrying_iou_thresh = getattr(self.cfg, 'carrying_iou_threshold', 0.1)
            far_dist = getattr(self.cfg, 'far_distance_px', 300)

            for person_id, pinfo in person_info.items():
                dist = distance(bag_center, pinfo['body_center'])

                if dist > far_dist:
                    continue

                # Check IoU overlap (bag inside person's body area)
                bag_iou = iou(bag.box, pinfo['body_box'])
                is_carrying = dist <= carrying_dist or bag_iou >= carrying_iou_thresh

                # Create observation
                obs = CandidateObservation(
                    person_id=person_id,
                    track_id=pinfo['track_id'],
                    timestamp=now,
                    distance=dist,
                    iou=bag_iou,
                    is_carrying=is_carrying
                )

                # Add to candidate history
                history = ownership.get_or_create_history(person_id)
                history.add_observation(obs)

                all_candidates.append({
                    'person_id': person_id,
                    'track_id': pinfo['track_id'],
                    'dist': dist,
                    'iou': bag_iou,
                    'is_carrying': is_carrying,
                    'history': history
                })

            # Use temporal voting to determine owner
            if all_candidates:
                ownership.last_any_person_nearby = now
                result = self._temporal_voting(ownership, all_candidates, now)
                if result:
                    self._apply_voting_result(ownership, result, bag_id, now, person_to_track)
            else:
                # No one nearby - apply confidence decay
                ownership.is_being_carried = False
                ownership.carrier_person_id = None
                self._apply_confidence_decay(ownership, now)

            # Update owner track_id from current mappings
            if ownership.owner_person_id and ownership.owner_person_id in person_to_track:
                ownership.owner_track_id = person_to_track[ownership.owner_person_id]

            # Assign to owner
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

    def _temporal_voting(
        self,
        ownership: BagOwnership,
        candidates: List[dict],
        now: float
    ) -> dict | None:
        """Apply temporal voting to select best owner.

        Scoring: 50% proximity + 30% temporal consistency + 20% IoU

        Returns the winning candidate or None.
        """
        if not candidates:
            return None

        # Calculate scores for each candidate
        scored_candidates = []
        max_dist = self.cfg.far_distance_px

        for cand in candidates:
            history = cand['history']

            # Proximity score (0-1, closer = better)
            proximity_score = 1.0 - min(cand['dist'] / max_dist, 1.0)

            # Temporal score (0-1, more consistent = better)
            window_s = self.cfg.history_window_frames / 30.0  # ~2 seconds
            temporal_score = history.get_temporal_score(now, window_s)

            # IoU score (0-1)
            iou_score = cand['iou']

            # Weighted combination
            total_score = (
                self.cfg.weight_proximity * proximity_score +
                self.cfg.weight_temporal * temporal_score +
                self.cfg.weight_iou * iou_score
            )

            scored_candidates.append({
                **cand,
                'proximity_score': proximity_score,
                'temporal_score': temporal_score,
                'iou_score': iou_score,
                'total_score': total_score
            })

        # Sort by total score
        scored_candidates.sort(key=lambda x: x['total_score'], reverse=True)

        # Check for ambiguity
        best = scored_candidates[0]
        ownership.is_ambiguous = False
        ownership.ambiguous_candidates = []

        if len(scored_candidates) > 1:
            second = scored_candidates[1]
            margin = best['total_score'] - second['total_score']

            if margin < self.cfg.ambiguity_margin:
                # Ambiguous situation
                ownership.is_ambiguous = True
                ownership.ambiguous_candidates = [
                    c['person_id'] for c in scored_candidates
                    if best['total_score'] - c['total_score'] < self.cfg.ambiguity_margin
                ]

                # In ambiguous situations, prefer existing owner if they're a candidate
                if ownership.owner_person_id in ownership.ambiguous_candidates:
                    for c in scored_candidates:
                        if c['person_id'] == ownership.owner_person_id:
                            return c

        return best

    def _apply_voting_result(
        self,
        ownership: BagOwnership,
        winner: dict,
        bag_id: str | None,
        now: float,
        person_to_track: Dict[str, int]
    ) -> None:
        """Apply the voting result to update ownership."""
        person_id = winner['person_id']
        track_id = winner['track_id']
        is_carrying = winner['is_carrying']

        ownership.is_being_carried = is_carrying
        ownership.carrier_person_id = person_id if is_carrying else None

        # Reset decay since owner is visible
        ownership.decay_start_time = None

        min_val_frames = getattr(self.cfg, 'minimum_validation_frames', 15)
        min_val_conf = getattr(self.cfg, 'minimum_validation_confidence', 0.6)
        transfer_confirm = getattr(self.cfg, 'transfer_confirmation_frames', 30)

        if ownership.owner_person_id is None:
            # First-time ownership assignment
            ownership.owner_person_id = person_id
            ownership.owner_track_id = track_id
            # Dynamic initial confidence based on carrying and score
            base_conf = 0.4 + (0.2 * winner['total_score'])
            if is_carrying:
                base_conf += 0.1
            ownership.confidence = base_conf
            ownership.base_confidence = base_conf
            ownership.link_frames = 1
            ownership.last_owner_seen = now
            ownership.validation_frames = 1
            logger.info(f"Bag {bag_id} assigned to owner {person_id} (conf={ownership.confidence:.2f})")

            self._add_bag_to_person(person_id, track_id, bag_id, ownership.bag_track_id)

        elif ownership.owner_person_id == person_id:
            # Same owner - increase confidence
            ownership.link_frames += 1
            ownership.last_owner_seen = now

            # Confidence growth curve
            growth = 0.02 + (0.01 * winner['total_score'])
            if is_carrying:
                growth += 0.01
            ownership.confidence = min(1.0, ownership.confidence + growth)
            ownership.base_confidence = ownership.confidence
            ownership.validation_frames += 1

            # Validate after sufficient frames
            if (ownership.validation_frames >= min_val_frames and
                ownership.confidence >= min_val_conf):
                ownership.validated = True

            # Reset transfer state
            ownership.transfer_candidate_id = None
            ownership.transfer_frames = 0

        else:
            # Different person - handle potential transfer
            if not ownership.is_ambiguous and is_carrying:
                # Only consider transfer if not ambiguous and new person is carrying
                if ownership.transfer_candidate_id == person_id:
                    ownership.transfer_frames += 1

                    if ownership.transfer_frames >= transfer_confirm:
                        # Transfer confirmed
                        old_owner = ownership.owner_person_id
                        ownership.owner_person_id = person_id
                        ownership.owner_track_id = track_id
                        ownership.confidence = 0.5
                        ownership.base_confidence = 0.5
                        ownership.validation_frames = 0
                        ownership.validated = False
                        ownership.transfer_candidate_id = None
                        ownership.transfer_frames = 0
                        ownership.last_owner_seen = now

                        if old_owner:
                            self._remove_bag_from_person(old_owner, bag_id)
                        self._add_bag_to_person(person_id, track_id, bag_id, ownership.bag_track_id)
                        logger.info(f"Bag {bag_id} ownership transferred: {old_owner} -> {person_id}")
                else:
                    # New transfer candidate
                    ownership.transfer_candidate_id = person_id
                    ownership.transfer_frames = 1
            else:
                # Ambiguous or not carrying - reset transfer
                ownership.transfer_candidate_id = None
                ownership.transfer_frames = 0

    def _apply_confidence_decay(self, ownership: BagOwnership, now: float) -> None:
        """Apply confidence decay when owner is not visible.

        Uses exponential decay: confidence = base * exp(-rate * time)
        """
        if ownership.owner_person_id is None:
            return

        # Start decay timer if not already started
        if ownership.decay_start_time is None:
            ownership.decay_start_time = now
            ownership.base_confidence = ownership.confidence
            return

        # Calculate decay
        time_since_seen = now - ownership.decay_start_time
        decay_rate = getattr(self.cfg, 'confidence_decay_rate', 0.05)
        min_floor = getattr(self.cfg, 'min_confidence_floor', 0.1)

        decayed = ownership.base_confidence * math.exp(-decay_rate * time_since_seen)
        ownership.confidence = max(min_floor, decayed)

    def _update_carrying_ownership(self, ownership: BagOwnership, candidate: dict, bag_id: str, now: float):
        """Update ownership when someone is actively carrying the bag."""
        person_id = candidate['person_id']
        track_id = candidate['track_id']
        iou_score = candidate['iou']

        ownership.is_being_carried = True
        ownership.carrier_person_id = person_id
        ownership.last_any_person_nearby = now

        min_val_frames = getattr(self.cfg, 'minimum_validation_frames', 15)
        min_val_conf = getattr(self.cfg, 'minimum_validation_confidence', 0.6)
        transfer_confirm = getattr(self.cfg, 'transfer_confirmation_frames', 30)

        if ownership.owner_person_id is None:
            # First-time ownership assignment
            ownership.owner_person_id = person_id
            ownership.owner_track_id = track_id
            # Dynamic initial confidence based on IoU
            ownership.confidence = 0.4 + (0.2 * iou_score)
            ownership.link_frames = 1
            ownership.last_owner_seen = now
            ownership.validation_frames = 1
            logger.info(f"Bag {bag_id} assigned to owner {person_id} (conf={ownership.confidence:.2f})")

            self._add_bag_to_person(person_id, track_id, bag_id, ownership.bag_track_id)

        elif ownership.owner_person_id == person_id:
            # Same owner - increase confidence
            ownership.link_frames += 1
            ownership.last_owner_seen = now
            ownership.owner_track_id = track_id  # Update track ID in case of re-id

            # Confidence growth curve (faster when IoU is high)
            ownership.confidence = min(1.0, ownership.confidence + 0.02 + 0.01 * iou_score)
            ownership.validation_frames += 1

            # Validate after sufficient frames
            if (ownership.validation_frames >= min_val_frames and
                ownership.confidence >= min_val_conf):
                ownership.validated = True

            # Reset transfer state
            ownership.transfer_candidate_id = None
            ownership.transfer_frames = 0

        else:
            # Different person - handle potential transfer
            if ownership.transfer_candidate_id == person_id:
                ownership.transfer_frames += 1

                if ownership.transfer_frames >= transfer_confirm:
                    # Transfer confirmed
                    old_owner = ownership.owner_person_id
                    ownership.owner_person_id = person_id
                    ownership.owner_track_id = track_id
                    ownership.confidence = 0.5  # Reset confidence for new owner
                    ownership.validation_frames = 0
                    ownership.validated = False
                    ownership.transfer_candidate_id = None
                    ownership.transfer_frames = 0
                    ownership.last_owner_seen = now

                    self._remove_bag_from_person(old_owner, bag_id)
                    self._add_bag_to_person(person_id, track_id, bag_id, ownership.bag_track_id)
                    logger.info(f"Bag {bag_id} ownership transferred: {old_owner} -> {person_id}")
            else:
                # New transfer candidate
                ownership.transfer_candidate_id = person_id
                ownership.transfer_frames = 1

    def _update_near_ownership(self, ownership: BagOwnership, candidate: dict, bag_id: str, now: float):
        """Update ownership when someone is near but not carrying."""
        person_id = candidate['person_id']
        track_id = candidate['track_id']

        ownership.is_being_carried = False
        ownership.carrier_person_id = None
        ownership.last_any_person_nearby = now

        if ownership.owner_person_id is None:
            # First-time assignment (lower confidence since not carrying)
            ownership.owner_person_id = person_id
            ownership.owner_track_id = track_id
            ownership.confidence = 0.3  # Lower than carrying
            ownership.link_frames = 1
            ownership.last_owner_seen = now
            ownership.validation_frames = 1
            logger.info(f"Bag {bag_id} tentatively assigned to {person_id} (near)")

            self._add_bag_to_person(person_id, track_id, bag_id, ownership.bag_track_id)

        elif ownership.owner_person_id == person_id:
            # Owner is near - slower confidence increase
            ownership.link_frames += 1
            ownership.last_owner_seen = now
            ownership.confidence = min(1.0, ownership.confidence + 0.01)
            ownership.validation_frames += 1

        # Don't transfer ownership based on "near" alone - require carrying

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

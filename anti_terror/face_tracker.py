"""Face-specific tracking and re-identification module.

This module provides robust face tracking with embedding-based re-identification
to minimize duplicate ID creation.

Key features:
- IdentityRegistry for track-to-person mapping with relink after occlusion
- FaceGallery with memory bounds and LRU eviction
- Temporal consistency for stable ID assignment
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
import time

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from supervision import Detections
from supervision.tracker.byte_tracker.core import ByteTrack

from .config import EmbeddingConfig, IdentityConfig, TrackingConfig


@dataclass
class FaceTrack:
    """Represents a tracked face with its embedding history."""
    track_id: int
    box: np.ndarray  # xyxy face bounding box
    score: float  # detection confidence

    # Identity
    person_id: str | None = None

    # Embedding state
    embedding: torch.Tensor | None = None
    embedding_history: list = field(default_factory=list)
    quality_history: list = field(default_factory=list)

    # Tracking state
    frames_seen: int = 0
    frames_since_embedding: int = 0
    last_seen_time: float = 0.0

    # Re-ID state
    candidate_id: str | None = None
    candidate_count: int = 0


class FaceGallery:
    """Gallery of known face identities with robust matching.

    Uses multiple strategies to minimize false new IDs:
    1. Centroid-based matching with quality weighting
    2. Temporal consistency checks
    3. Adaptive thresholds based on gallery size
    4. Memory bounds via IdentityConfig
    """

    def __init__(self, cfg: EmbeddingConfig, identity_cfg: IdentityConfig | None = None):
        self.cfg = cfg
        self.identity_cfg = identity_cfg or IdentityConfig()
        self.identities: Dict[str, FaceIdentity] = {}
        self.counter = 0
        self.last_creation_time = 0.0

    def _new_id(self) -> str:
        self.counter += 1
        return f"P_{self.counter:04d}"

    def match(self, embedding: torch.Tensor, quality: float = 1.0) -> Tuple[str | None, float]:
        """Find best matching identity for an embedding.

        Returns (identity_id, similarity_score) or (None, 0.0) if no match.
        """
        if not self.identities:
            return None, 0.0

        best_id = None
        best_score = -1.0

        for pid, identity in self.identities.items():
            score = identity.match_score(embedding)
            if score > best_score:
                best_score = score
                best_id = pid

        return best_id, best_score

    def match_or_create(
        self,
        embedding: torch.Tensor,
        quality: float,
        face_crop: np.ndarray | None = None,
        min_quality_for_create: float = 0.4
    ) -> Tuple[str, bool, float]:
        """Match embedding to existing identity or create new one.

        Very conservative about creating new IDs - multiple fallback checks.
        Returns (identity_id, was_created, match_score).
        """
        best_id, best_score = self.match(embedding, quality)

        # Adaptive threshold based on gallery size
        # More IDs = stricter matching to avoid duplicates
        base_threshold = self.cfg.face_similarity_threshold
        if len(self.identities) > 5:
            # Lower threshold when many IDs exist (be more permissive in matching)
            adaptive_threshold = base_threshold - 0.05 * min(len(self.identities) - 5, 5) / 5
        else:
            adaptive_threshold = base_threshold

        # Primary match
        if best_id and best_score >= adaptive_threshold:
            self.identities[best_id].add_embedding(embedding, quality)
            return best_id, False, best_score

        # Secondary match (force threshold)
        force_threshold = adaptive_threshold - self.cfg.face_force_match_margin
        if best_id and best_score >= force_threshold:
            self.identities[best_id].add_embedding(embedding, quality)
            return best_id, False, best_score

        # Tertiary match (very low threshold to avoid duplicates)
        min_threshold = 0.25
        if best_id and best_score >= min_threshold:
            logger.debug(f"Low-confidence match ({best_score:.3f}) to {best_id}")
            self.identities[best_id].add_embedding(embedding, quality)
            return best_id, False, best_score

        # Check cooldown before creating
        now = time.time()
        if (now - self.last_creation_time) < self.cfg.face_new_id_cooldown_s:
            if best_id:
                # Cooldown active - reuse best match anyway
                self.identities[best_id].add_embedding(embedding, quality)
                return best_id, False, best_score
            # No match and cooldown active - return None
            return "", False, 0.0

        # Quality check for new ID
        if quality < min_quality_for_create:
            if best_id:
                return best_id, False, best_score
            return "", False, 0.0

        # Create new identity
        new_id = self._new_id()
        self.identities[new_id] = FaceIdentity(new_id, embedding, quality)
        self.last_creation_time = now
        logger.info(f"Created new identity {new_id} (best_match: {best_id}, score: {best_score:.3f})")
        return new_id, True, 1.0

    def get_all_ids(self) -> List[str]:
        return list(self.identities.keys())


@dataclass
class FaceIdentity:
    """Represents a known face identity with embedding history."""
    identity_id: str
    embeddings: list = field(default_factory=list)
    qualities: list = field(default_factory=list)
    centroid: torch.Tensor | None = None
    created_time: float = field(default_factory=time.time)
    last_seen_time: float = field(default_factory=time.time)
    max_history: int = 30

    def __init__(self, identity_id: str, initial_embedding: torch.Tensor, quality: float = 1.0):
        self.identity_id = identity_id
        self.embeddings = [initial_embedding]
        self.qualities = [quality]
        self.centroid = initial_embedding.clone()
        self.created_time = time.time()
        self.last_seen_time = time.time()
        self.max_history = 30

    def add_embedding(self, embedding: torch.Tensor, quality: float = 1.0) -> None:
        """Add new embedding observation."""
        self.embeddings.append(embedding)
        self.qualities.append(quality)
        self.last_seen_time = time.time()

        # Trim history (keep highest quality)
        if len(self.embeddings) > self.max_history:
            # Sort by quality and keep best
            paired = list(zip(self.embeddings, self.qualities))
            paired.sort(key=lambda x: x[1], reverse=True)
            self.embeddings = [p[0] for p in paired[:self.max_history]]
            self.qualities = [p[1] for p in paired[:self.max_history]]

        self._update_centroid()

    def _update_centroid(self) -> None:
        """Recompute quality-weighted centroid."""
        if not self.embeddings:
            return

        weights = torch.tensor(self.qualities)
        weights = weights / weights.sum()

        stacked = torch.stack(self.embeddings)
        weighted_sum = (stacked * weights.unsqueeze(1)).sum(dim=0)
        self.centroid = F.normalize(weighted_sum, dim=0)

    def match_score(self, embedding: torch.Tensor) -> float:
        """Compute similarity score for an embedding."""
        if self.centroid is None:
            return 0.0

        # Primary: centroid similarity
        centroid_sim = float(F.cosine_similarity(
            embedding.unsqueeze(0),
            self.centroid.unsqueeze(0)
        ).item())

        # Secondary: best recent match
        recent = self.embeddings[-5:] if len(self.embeddings) >= 5 else self.embeddings
        recent_scores = [
            float(F.cosine_similarity(embedding.unsqueeze(0), e.unsqueeze(0)).item())
            for e in recent
        ]
        recent_max = max(recent_scores) if recent_scores else 0.0

        # Combined: 60% centroid + 40% recent (more weight to recent for temporal consistency)
        return 0.6 * centroid_sim + 0.4 * recent_max


@dataclass
class TrackHistory:
    """History of a track_id for a person."""
    track_id: int
    start_time: float
    end_time: float
    embedding: torch.Tensor | None = None


class IdentityRegistry:
    """Registry for managing track_id to person_id mappings.

    Solves the critical problem of track ID loss during occlusion:
    - Maintains bidirectional mapping between track_ids and person_ids
    - Supports relinking after ByteTrack loses a track
    - Implements LRU eviction for memory bounds
    """

    def __init__(self, cfg: IdentityConfig, gallery: 'FaceGallery'):
        self.cfg = cfg
        self.gallery = gallery

        # Bidirectional mappings
        self.track_to_person: Dict[int, str] = {}  # track_id -> person_id
        self.person_to_tracks: Dict[str, Set[int]] = {}  # person_id -> set of track_ids

        # Track history for relink (stores recent lost tracks)
        self.lost_tracks: Dict[int, TrackHistory] = {}  # track_id -> history

        # LRU tracking for eviction
        self.person_last_seen: Dict[str, float] = {}  # person_id -> timestamp

        logger.info("IdentityRegistry initialized")

    def register_track(self, track_id: int, person_id: str, embedding: torch.Tensor | None = None) -> None:
        """Register a track_id to person_id mapping."""
        now = time.time()

        # Update mappings
        self.track_to_person[track_id] = person_id

        if person_id not in self.person_to_tracks:
            self.person_to_tracks[person_id] = set()
        self.person_to_tracks[person_id].add(track_id)

        # Update LRU
        self.person_last_seen[person_id] = now

    def get_person_id(self, track_id: int) -> str | None:
        """Get person_id for a track_id."""
        return self.track_to_person.get(track_id)

    def get_track_ids(self, person_id: str) -> Set[int]:
        """Get all track_ids for a person_id."""
        return self.person_to_tracks.get(person_id, set()).copy()

    def mark_track_lost(self, track_id: int, embedding: torch.Tensor | None = None) -> None:
        """Mark a track as lost (for potential relink later)."""
        person_id = self.track_to_person.get(track_id)
        if not person_id:
            return

        now = time.time()
        self.lost_tracks[track_id] = TrackHistory(
            track_id=track_id,
            start_time=now,
            end_time=now,
            embedding=embedding
        )

        # Don't remove from mappings yet - keep for relink window
        logger.debug(f"Track {track_id} ({person_id}) marked as lost")

    def try_relink(self, new_track_id: int, embedding: torch.Tensor) -> str | None:
        """Try to relink a new track_id to a recently lost person.

        Returns person_id if relink successful, None otherwise.
        """
        now = time.time()
        best_person = None
        best_score = 0.0

        # Check all lost tracks within relink window
        for lost_track_id, history in list(self.lost_tracks.items()):
            if now - history.start_time > self.cfg.relink_window_s:
                # Expired - remove
                self._cleanup_lost_track(lost_track_id)
                continue

            person_id = self.track_to_person.get(lost_track_id)
            if not person_id:
                continue

            # Compare with gallery identity
            if person_id in self.gallery.identities:
                score = self.gallery.identities[person_id].match_score(embedding)
                if score >= self.cfg.relink_threshold and score > best_score:
                    best_score = score
                    best_person = person_id

        if best_person:
            logger.info(f"Relinked track {new_track_id} to {best_person} (score: {best_score:.3f})")
            self.register_track(new_track_id, best_person, embedding)
            return best_person

        return None

    def _cleanup_lost_track(self, track_id: int) -> None:
        """Cleanup a lost track from registry."""
        if track_id in self.lost_tracks:
            del self.lost_tracks[track_id]

        # Optionally remove from track_to_person if truly stale
        # (but keep person_to_tracks for identity persistence)

    def update_seen(self, person_id: str) -> None:
        """Update last seen time for a person."""
        self.person_last_seen[person_id] = time.time()

    def evict_stale_identities(self) -> List[str]:
        """Evict stale identities to maintain memory bounds.

        Uses LRU eviction based on identity_ttl_hours.
        Returns list of evicted person_ids.
        """
        now = time.time()
        ttl_seconds = self.cfg.identity_ttl_hours * 3600
        evicted = []

        # Check for TTL-based eviction
        for person_id, last_seen in list(self.person_last_seen.items()):
            if now - last_seen > ttl_seconds:
                self._evict_person(person_id)
                evicted.append(person_id)

        # Check for size-based eviction
        while len(self.gallery.identities) > self.cfg.max_gallery_size:
            # Find LRU person
            if not self.person_last_seen:
                break

            oldest_person = min(self.person_last_seen.keys(), key=lambda p: self.person_last_seen[p])
            self._evict_person(oldest_person)
            evicted.append(oldest_person)

        if evicted:
            logger.info(f"Evicted {len(evicted)} stale identities: {evicted}")

        return evicted

    def _evict_person(self, person_id: str) -> None:
        """Evict a person from all registries."""
        # Remove from gallery
        if person_id in self.gallery.identities:
            del self.gallery.identities[person_id]

        # Remove track mappings
        if person_id in self.person_to_tracks:
            for track_id in self.person_to_tracks[person_id]:
                if track_id in self.track_to_person:
                    del self.track_to_person[track_id]
            del self.person_to_tracks[person_id]

        # Remove LRU entry
        if person_id in self.person_last_seen:
            del self.person_last_seen[person_id]

    def cleanup_expired_lost_tracks(self) -> None:
        """Remove expired lost tracks."""
        now = time.time()
        expired = [
            tid for tid, history in self.lost_tracks.items()
            if now - history.start_time > self.cfg.relink_window_s
        ]
        for tid in expired:
            self._cleanup_lost_track(tid)

    def get_stats(self) -> dict:
        """Get registry statistics."""
        return {
            'active_tracks': len(self.track_to_person),
            'known_persons': len(self.person_to_tracks),
            'lost_tracks': len(self.lost_tracks),
            'gallery_size': len(self.gallery.identities),
        }


class FaceTracker:
    """Specialized tracker for faces with embedding-based re-identification.

    Key features:
    1. ByteTrack for spatial tracking
    2. Embedding matching for re-identification
    3. IdentityRegistry for track-to-person mapping and relink
    4. Temporal consistency for stable IDs
    """

    def __init__(
        self,
        tracking_cfg: TrackingConfig,
        embedding_cfg: EmbeddingConfig,
        identity_cfg: IdentityConfig | None = None
    ):
        self.tracking_cfg = tracking_cfg
        self.embedding_cfg = embedding_cfg
        self.identity_cfg = identity_cfg or IdentityConfig()

        # Spatial tracker (ByteTrack)
        self.byte_tracker = ByteTrack(
            track_activation_threshold=0.5,  # Higher for faces
            lost_track_buffer=tracking_cfg.lost_track_buffer,
            minimum_matching_threshold=0.7,
            frame_rate=tracking_cfg.frame_rate,
            minimum_consecutive_frames=2,
        )

        # Identity gallery
        self.gallery = FaceGallery(embedding_cfg, self.identity_cfg)

        # Identity registry for track-to-person mapping
        self.registry = IdentityRegistry(self.identity_cfg, self.gallery)

        # Active tracks
        self.tracks: Dict[int, FaceTrack] = {}

        # Previous frame's active track IDs (for detecting lost tracks)
        self._prev_active_tracks: Set[int] = set()

        # Frame counter for periodic cleanup
        self._frame_count = 0

        logger.info("Initialized FaceTracker with IdentityRegistry")

    def update(
        self,
        face_boxes: np.ndarray,
        face_scores: np.ndarray,
        face_embeddings: List[torch.Tensor],
        face_qualities: List[float],
        face_crops: List[np.ndarray]
    ) -> List[FaceTrack]:
        """Update tracker with new face detections.

        Args:
            face_boxes: Nx4 array of face bounding boxes (xyxy)
            face_scores: N detection confidence scores
            face_embeddings: N face embeddings
            face_qualities: N quality scores
            face_crops: N face crop images

        Returns:
            List of active FaceTrack objects with assigned person_ids
        """
        current_time = time.time()
        self._frame_count += 1

        # Handle empty detections
        if len(face_boxes) == 0:
            self.byte_tracker.update_with_detections(Detections.empty())
            # Mark all current tracks as lost
            for tid in self._prev_active_tracks:
                if tid in self.tracks:
                    self.registry.mark_track_lost(tid, self.tracks[tid].embedding)
            # Age out old tracks
            for tid in list(self.tracks.keys()):
                if current_time - self.tracks[tid].last_seen_time > 2.0:
                    del self.tracks[tid]
            self._prev_active_tracks = set()
            return []

        # Run ByteTrack
        detections = Detections(
            xyxy=face_boxes,
            confidence=face_scores,
            class_id=np.zeros(len(face_boxes), dtype=int),  # All faces = class 0
        )
        tracked = self.byte_tracker.update_with_detections(detections)

        result: List[FaceTrack] = []
        active_track_ids: Set[int] = set()

        for idx, track_id in enumerate(tracked.tracker_id):
            track_id = int(track_id)
            active_track_ids.add(track_id)

            box = tracked.xyxy[idx]
            score = float(tracked.confidence[idx])

            # Find corresponding detection (closest box)
            det_idx = self._find_detection_idx(box, face_boxes)
            if det_idx is None or det_idx >= len(face_embeddings):
                continue

            embedding = face_embeddings[det_idx]
            quality = face_qualities[det_idx]
            crop = face_crops[det_idx] if det_idx < len(face_crops) else None

            # Get or create track state
            is_new_track = track_id not in self.tracks
            if is_new_track:
                self.tracks[track_id] = FaceTrack(
                    track_id=track_id,
                    box=box,
                    score=score
                )

            track = self.tracks[track_id]
            track.box = box
            track.score = score
            track.frames_seen += 1
            track.last_seen_time = current_time

            # Update embedding (EMA smoothing)
            if track.embedding is None:
                track.embedding = embedding
            else:
                alpha = self.embedding_cfg.ema_alpha
                # Quality-weighted alpha
                if self.embedding_cfg.ema_quality_weighted:
                    q_factor = min(quality / 0.6, 1.0)
                    new_weight = (1 - alpha) * (0.5 + 0.5 * q_factor)
                else:
                    new_weight = 1 - alpha
                track.embedding = F.normalize(
                    (1 - new_weight) * track.embedding + new_weight * embedding,
                    dim=0
                )

            track.embedding_history.append(embedding)
            track.quality_history.append(quality)
            if len(track.embedding_history) > 10:
                track.embedding_history = track.embedding_history[-10:]
                track.quality_history = track.quality_history[-10:]

            # Assign person_id
            if track.person_id is None:
                # New track - first try to relink to recently lost person
                if is_new_track:
                    relinked_pid = self.registry.try_relink(track_id, embedding)
                    if relinked_pid:
                        track.person_id = relinked_pid
                        # Update gallery with new observation
                        if relinked_pid in self.gallery.identities:
                            self.gallery.identities[relinked_pid].add_embedding(embedding, quality)

                # If not relinked, try normal matching/creation
                if track.person_id is None:
                    if track.frames_seen >= self.embedding_cfg.face_new_id_patience_frames:
                        # Enough frames seen - try to match or create
                        pid, created, match_score = self.gallery.match_or_create(
                            track.embedding,
                            quality=max(track.quality_history) if track.quality_history else quality,
                            face_crop=crop,
                            min_quality_for_create=self.embedding_cfg.min_face_quality
                        )
                        if pid:
                            track.person_id = pid
                            self.registry.register_track(track_id, pid, embedding)
                    else:
                        # Not enough frames - try to match existing only
                        best_id, best_score = self.gallery.match(track.embedding, quality)
                        if best_id and best_score >= self.embedding_cfg.face_similarity_threshold - 0.1:
                            track.person_id = best_id
                            self.registry.register_track(track_id, best_id, embedding)
                            self.gallery.identities[best_id].add_embedding(embedding, quality)
            else:
                # Existing track - verify and update
                current_id = track.person_id
                best_id, best_score = self.gallery.match(track.embedding, quality)

                # Update registry LRU
                self.registry.update_seen(current_id)

                # Update gallery with new observation
                if current_id in self.gallery.identities:
                    self.gallery.identities[current_id].add_embedding(embedding, quality)

                # Check if should switch ID (requires consistency)
                if best_id and best_id != current_id:
                    threshold = self.embedding_cfg.face_similarity_threshold
                    if best_score >= threshold + 0.1:  # Significantly better match
                        if track.candidate_id == best_id:
                            track.candidate_count += 1
                            if track.candidate_count >= self.embedding_cfg.face_switch_patience_frames:
                                logger.info(f"Track {track_id}: switching {current_id} -> {best_id}")
                                track.person_id = best_id
                                self.registry.register_track(track_id, best_id, embedding)
                                track.candidate_id = None
                                track.candidate_count = 0
                        else:
                            track.candidate_id = best_id
                            track.candidate_count = 1
                    else:
                        track.candidate_id = None
                        track.candidate_count = 0

            result.append(track)

        # Detect lost tracks (were active last frame, not active now)
        lost_tracks = self._prev_active_tracks - active_track_ids
        for tid in lost_tracks:
            if tid in self.tracks:
                self.registry.mark_track_lost(tid, self.tracks[tid].embedding)

        # Cleanup old tracks
        for tid in list(self.tracks.keys()):
            if tid not in active_track_ids:
                if current_time - self.tracks[tid].last_seen_time > 3.0:
                    del self.tracks[tid]

        # Periodic cleanup
        if self._frame_count % self.identity_cfg.cleanup_interval_frames == 0:
            self.registry.evict_stale_identities()
            self.registry.cleanup_expired_lost_tracks()

        self._prev_active_tracks = active_track_ids
        return result

    def _find_detection_idx(self, track_box: np.ndarray, det_boxes: np.ndarray) -> Optional[int]:
        """Find detection index that best matches track box."""
        if len(det_boxes) == 0:
            return None

        # Compute IoU between track box and all detections
        best_iou = 0.0
        best_idx = None

        for i, det_box in enumerate(det_boxes):
            iou = self._compute_iou(track_box, det_box)
            if iou > best_iou:
                best_iou = iou
                best_idx = i

        return best_idx if best_iou > 0.3 else None

    @staticmethod
    def _compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
        """Compute IoU between two boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter

        return inter / union if union > 0 else 0.0

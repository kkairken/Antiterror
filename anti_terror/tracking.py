from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from loguru import logger
from supervision import Detections
from supervision.tracker.byte_tracker.core import ByteTrack

from .config import TrackingConfig


@dataclass
class Track:
    track_id: int
    box: np.ndarray  # xyxy
    score: float
    cls: int
    frames_seen: int = 1  # How many frames this track has been observed


class Tracker:
    def __init__(self, cfg: TrackingConfig):
        self.cfg = cfg
        self.byte_tracker = ByteTrack(
            track_activation_threshold=cfg.track_activation_threshold,
            lost_track_buffer=cfg.lost_track_buffer,
            minimum_matching_threshold=cfg.minimum_matching_threshold,
            frame_rate=cfg.frame_rate,
            minimum_consecutive_frames=cfg.minimum_consecutive_frames,
        )
        # Track history: track_id -> frames_seen count
        self._track_frames: Dict[int, int] = {}
        logger.info("Initialized ByteTrack tracker")

    def update(self, detections_xyxy: np.ndarray, scores: np.ndarray, classes: np.ndarray) -> List[Track]:
        if len(detections_xyxy) == 0:
            self.byte_tracker.update_with_detections(Detections.empty())
            return []
        det = Detections(
            xyxy=detections_xyxy,
            confidence=scores,
            class_id=classes,
        )
        tracks = self.byte_tracker.update_with_detections(det)
        result: List[Track] = []
        active_ids = set()
        for idx, track_id in enumerate(tracks.tracker_id):
            tid = int(track_id)
            active_ids.add(tid)
            # Update frames count
            self._track_frames[tid] = self._track_frames.get(tid, 0) + 1
            result.append(
                Track(
                    track_id=tid,
                    box=tracks.xyxy[idx],
                    score=float(tracks.confidence[idx]),
                    cls=int(tracks.class_id[idx]),
                    frames_seen=self._track_frames[tid],
                )
            )
        # Cleanup old tracks not seen for a while (memory management)
        stale_ids = [tid for tid in self._track_frames if tid not in active_ids]
        for tid in stale_ids:
            if self._track_frames[tid] > self.cfg.lost_track_buffer * 2:
                del self._track_frames[tid]
        return result

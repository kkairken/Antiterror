"""Main pipeline for video processing with face-based tracking.

This module now uses direct face tracking instead of person-body tracking
for more accurate face recognition and reduced duplicate IDs.
"""
import argparse
import time
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
import torch
from loguru import logger
import threading
import http.server
import socketserver

from .association import AssociationEngine
from .behavior import BehaviorAnalyzer
from .config import PipelineConfig, select_device
from .database import Database
from .detection import Detector
from .embeddings import BagEmbedder, EmbeddingStore, FaceEmbedder, FaceQuality
from .events import EventSink
from .face_tracker import FaceTracker, FaceTrack
from .tracking import Tracker
from .video import open_video_source, read_frame, release


class PreviewServer:
    """Simple MJPEG preview server for headless environments."""

    def __init__(self, port: int):
        self.port = port
        self.latest_jpeg: Optional[bytes] = None
        self.lock = threading.Lock()

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self_inner):
                if self_inner.path != "/":
                    self_inner.send_response(404)
                    self_inner.end_headers()
                    return
                self_inner.send_response(200)
                self_inner.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self_inner.end_headers()
                try:
                    while True:
                        with self.lock:
                            jpeg = self.latest_jpeg
                        if jpeg is None:
                            time.sleep(0.05)
                            continue
                        self_inner.wfile.write(b"--frame\r\n")
                        self_inner.wfile.write(b"Content-Type: image/jpeg\r\n\r\n")
                        self_inner.wfile.write(jpeg)
                        self_inner.wfile.write(b"\r\n")
                        self_inner.wfile.flush()
                        time.sleep(0.03)
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def log_message(self_inner, format, *args):
                return  # silence

        self.httpd = socketserver.ThreadingTCPServer(("0.0.0.0", port), Handler)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        logger.info(f"Preview server started at http://localhost:{port}")

    def update(self, frame: np.ndarray):
        try:
            ret, buf = cv2.imencode(".jpg", frame)
            if ret:
                with self.lock:
                    self.latest_jpeg = buf.tobytes()
        except Exception as e:
            logger.warning(f"Preview encode failed: {e}")

    def stop(self):
        try:
            self.httpd.shutdown()
            self.httpd.server_close()
        except Exception:
            pass


class Pipeline:
    """Video processing pipeline with face-centric tracking.

    Key features:
    1. Faces are tracked directly (not extracted from person boxes)
    2. Face bounding boxes are drawn instead of body boxes
    3. FaceTracker handles re-identification with gallery matching
    4. Bags are linked to owners with persistent tracking
    5. Database stores all relationships
    """

    def __init__(self, cfg: PipelineConfig):
        # Adjust device globally
        cfg.detection.device = select_device(cfg.detection.device)
        cfg.embeddings.device = select_device(cfg.embeddings.device)

        self.cfg = cfg

        # Database for persistent storage
        self.db = Database("antiterror.db")

        # Object detection (for bags)
        self.detector = Detector(cfg.detection)
        self.tracker = Tracker(cfg.tracking)

        # Face detection and embedding
        self.face_embedder = FaceEmbedder(cfg.embeddings)

        # Face-specific tracker with re-identification
        self.face_tracker = FaceTracker(cfg.tracking, cfg.embeddings)

        # Bag handling
        self.bag_embedder = BagEmbedder(cfg.embeddings)
        self.bag_store = EmbeddingStore()

        # Association and behavior analysis
        self.assoc = AssociationEngine(cfg.association)
        self.behavior = BehaviorAnalyzer(cfg.behavior)
        self.events = EventSink(cfg.events)

        # Video source
        self.cap = open_video_source(cfg.video_source)

        # Store face boxes for association with bags
        self.current_face_tracks: List[FaceTrack] = []
        self.render_enabled: bool = True
        self.preview: Optional[PreviewServer] = None

        # Frame counter for periodic DB updates
        self.frame_count = 0

        logger.info("Pipeline initialized with face-centric tracking and database")

    def process_frame(self, frame: np.ndarray) -> None:
        """Process a single video frame.

        Pipeline:
        1. Detect faces and extract embeddings
        2. Track faces with FaceTracker (handles re-identification)
        3. Detect and track bags
        4. Associate bags with face owners
        5. Update database
        6. Render visualization
        """
        self.frame_count += 1

        # === Face Detection and Tracking ===
        face_detections = self.face_embedder(frame)

        if face_detections:
            # Unpack face detections
            face_boxes = np.array([fd[0] for fd in face_detections])
            face_scores = np.array([fd[1] for fd in face_detections])
            face_embeddings = [fd[2] for fd in face_detections]
            face_crops = [fd[3] for fd in face_detections]
            face_qualities = [fd[4].overall_quality for fd in face_detections]

            # Update face tracker
            self.current_face_tracks = self.face_tracker.update(
                face_boxes=face_boxes,
                face_scores=face_scores,
                face_embeddings=face_embeddings,
                face_qualities=face_qualities,
                face_crops=face_crops
            )
        else:
            # No faces detected - update tracker with empty
            self.current_face_tracks = self.face_tracker.update(
                face_boxes=np.array([]).reshape(0, 4),
                face_scores=np.array([]),
                face_embeddings=[],
                face_qualities=[],
                face_crops=[]
            )

        # === Bag Detection and Tracking ===
        detection = self.detector(frame)
        tracks = self.tracker.update(detection.boxes, detection.scores, detection.classes)
        bag_tracks = [t for t in tracks if t.cls in self.cfg.detection.classes_bag]

        # Assign bag IDs via embeddings
        bag_ids: Dict[int, str] = {}
        for bag in bag_tracks:
            x1, y1, x2, y2 = bag.box.astype(int)
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            emb = self.bag_embedder(crop)
            label, created, _ = self.bag_store.match_or_create(
                emb,
                prefix="B",
                threshold=self.cfg.embeddings.bag_similarity_threshold,
                image=cv2.cvtColor(crop, cv2.COLOR_BGR2RGB),
                save_dir=self.cfg.persistence.bags_dir if self.cfg.persistence.save_bags else None,
                strict_mode=True,  # Bags need separate IDs, not aggressive merging
            )
            bag_ids[bag.track_id] = label

        # === Association ===
        # Create person_ids dict from face tracks
        person_ids: Dict[int, str] = {}
        for ft in self.current_face_tracks:
            if ft.person_id:
                person_ids[ft.track_id] = ft.person_id

        # Use improved association engine
        assignments = self.assoc.associate(
            face_tracks=self.current_face_tracks,
            bag_tracks=bag_tracks,
            person_ids=person_ids,
            bag_ids=bag_ids
        )

        # === Update Database (periodically) ===
        if self.frame_count % 30 == 0:  # Every ~1 second at 30fps
            self._update_database(person_ids, bag_ids, assignments)

        # === Behavior Analysis ===
        events = self.behavior.update(bag_tracks, bag_ids, person_ids, assignments)
        if events:
            self.events.emit(events)
            # Log events to database
            for event in events:
                self.db.log_event(
                    event_type=event.get('type', 'unknown'),
                    bag_id=event.get('bag_id'),
                    person_id=event.get('person_id'),
                    details=event
                )

        # === Render ===
        self._render(frame, self.current_face_tracks, bag_tracks, bag_ids, assignments)

    def _update_database(
        self,
        person_ids: Dict[int, str],
        bag_ids: Dict[int, str],
        assignments: Dict[int, int]
    ) -> None:
        """Update database with current state."""
        # Update persons
        for track_id, person_id in person_ids.items():
            self.db.add_person(person_id)

        # Update bags and ownership
        for bag_track_id, bag_id in bag_ids.items():
            owner_person_id = self.assoc.get_bag_owner(bag_track_id)
            self.db.add_bag(bag_id, owner_person_id)

            if owner_person_id:
                confidence = self.assoc.get_ownership_confidence(bag_track_id)
                self.db.link_bag_to_person(bag_id, owner_person_id, confidence)

    def _render(
        self,
        frame: np.ndarray,
        face_tracks: List[FaceTrack],
        bag_tracks: List,
        bag_ids: Dict[int, str],
        assignments: Dict[int, int]
    ) -> None:
        """Render face boxes and bag boxes on frame."""
        # Draw face boxes
        for ft in face_tracks:
            x1, y1, x2, y2 = map(int, ft.box)

            # Color based on ID status
            if ft.person_id:
                color = (0, 255, 0)  # Green - has ID
            else:
                color = (0, 255, 255)  # Yellow - pending ID

            # Draw face rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Draw label with bag count
            label = ft.person_id if ft.person_id else f"?{ft.track_id}"
            if ft.person_id:
                bags = self.assoc.get_person_bags(ft.person_id)
                if bags:
                    label += f" [{len(bags)}]"  # Show bag count

            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]

            # Background for text
            cv2.rectangle(
                frame,
                (x1, y1 - label_size[1] - 10),
                (x1 + label_size[0] + 5, y1),
                color,
                -1
            )
            cv2.putText(
                frame,
                label,
                (x1 + 2, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),  # Black text on colored background
                2
            )

        # Draw bag boxes with owner info
        for bag in bag_tracks:
            x1, y1, x2, y2 = map(int, bag.box)

            bag_id = bag_ids.get(bag.track_id, f"B{bag.track_id}")
            owner_person_id = self.assoc.get_bag_owner(bag.track_id)
            is_carried = self.assoc.is_bag_being_carried(bag.track_id)
            confidence = self.assoc.get_ownership_confidence(bag.track_id)

            # Color based on state
            if is_carried:
                color = (0, 255, 0)  # Green - being carried
            elif owner_person_id:
                color = (0, 165, 255)  # Orange - has owner but not carried
            else:
                color = (0, 0, 255)  # Red - no owner

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Build label
            owner_label = owner_person_id if owner_person_id else "?"
            conf_str = f"{confidence:.0%}" if owner_person_id else ""
            text = f"{bag_id}->{owner_label} {conf_str}"

            # Label background
            label_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(
                frame,
                (x1, y1 - label_size[1] - 8),
                (x1 + label_size[0] + 4, y1),
                color,
                -1
            )
            cv2.putText(
                frame,
                text,
                (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                2
            )

        # Show stats
        num_faces = len(face_tracks)
        num_ids = len(self.face_tracker.gallery.get_all_ids())
        num_bags = len(bag_tracks)
        ownerships = self.assoc.get_all_ownerships()
        num_owned = len(ownerships)

        stats_text = f"Faces: {num_faces} | IDs: {num_ids} | Bags: {num_bags} | Owned: {num_owned}"
        cv2.putText(frame, stats_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Show ownership summary
        y_offset = 55
        for bag_id, person_id in list(ownerships.items())[:5]:  # Show top 5
            cv2.putText(
                frame,
                f"{bag_id} -> {person_id}",
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200, 200, 200),
                1
            )
            y_offset += 20

        if self.preview:
            self.preview.update(frame)

        if self.render_enabled:
            try:
                cv2.imshow("AntiTerror - Face & Bag Tracking", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    raise KeyboardInterrupt
            except Exception as e:
                logger.warning(f"Disabling rendering due to OpenCV GUI error: {e}")
                self.render_enabled = False

    def run(self) -> None:
        """Run the pipeline on video source."""
        logger.info("Starting face-centric pipeline. Press 'q' to exit.")
        try:
            while True:
                frame = read_frame(self.cap)
                if frame is None:
                    break
                self.process_frame(frame)
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            release(self.cap)
            if self.render_enabled:
                try:
                    cv2.destroyAllWindows()
                except Exception as e:
                    logger.warning(f"cv2.destroyAllWindows failed: {e}")
            if self.preview:
                self.preview.stop()

            # Log final stats
            num_ids = len(self.face_tracker.gallery.get_all_ids())
            db_stats = self.db.get_stats()
            logger.info(f"Final stats: {num_ids} faces, {db_stats}")

            # Close database
            self.db.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Anti-terror video analytics with face tracking")
    parser.add_argument("--source", type=str, default="0", help="Video source (index or path)")
    parser.add_argument("--camera-id", type=str, default="CAM_01", help="Camera identifier")
    parser.add_argument("--conf", type=float, default=None, help="Detection confidence override")
    parser.add_argument("--abandonment-timeout", type=float, default=None, help="Seconds to flag abandoned bag")
    parser.add_argument("--db", type=str, default="antiterror.db", help="Database file path")
    parser.add_argument("--preview-port", type=int, default=None, help="Start MJPEG preview server on this port")
    return parser.parse_args()


def main():
    args = parse_args()
    source: str | int = int(args.source) if args.source.isdigit() else args.source
    cfg = PipelineConfig(video_source=source)
    cfg.events.camera_id = args.camera_id
    if args.conf is not None:
        cfg.detection.conf_threshold = args.conf
    if args.abandonment_timeout is not None:
        cfg.behavior.abandonment_timeout_s = args.abandonment_timeout

    pipeline = Pipeline(cfg)
    if args.preview_port:
        pipeline.preview = PreviewServer(args.preview_port)
        pipeline.render_enabled = False  # disable GUI render if serving over HTTP
    pipeline.run()


if __name__ == "__main__":
    main()

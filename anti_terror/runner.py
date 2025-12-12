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
                return  # silence default logging

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

    Key changes from person-body tracking:
    1. Faces are tracked directly (not extracted from person boxes)
    2. Face bounding boxes are drawn instead of body boxes
    3. FaceTracker handles re-identification with gallery matching
    """

    def __init__(self, cfg: PipelineConfig):
        # Adjust device globally
        cfg.detection.device = select_device(cfg.detection.device)
        cfg.embeddings.device = select_device(cfg.embeddings.device)

        self.cfg = cfg

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

        # Behavior analysis
        self.assoc = AssociationEngine(cfg.association)
        self.behavior = BehaviorAnalyzer(cfg.behavior)
        self.events = EventSink(cfg.events)

        # Video source
        self.cap = open_video_source(cfg.video_source)

        # Store face boxes for association with bags
        self.current_face_tracks: List[FaceTrack] = []
        self.render_enabled: bool = True  # can be disabled on headless systems
        self.preview: Optional[PreviewServer] = None

        logger.info("Pipeline initialized with face-centric tracking")

    def process_frame(self, frame: np.ndarray) -> None:
        """Process a single video frame.

        Pipeline:
        1. Detect faces and extract embeddings
        2. Track faces with FaceTracker (handles re-identification)
        3. Detect and track bags (unchanged)
        4. Associate faces with bags for behavior analysis
        5. Render face boxes with IDs
        """
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
            )
            bag_ids[bag.track_id] = label

        # === Association and Behavior ===
        # Create person_ids dict from face tracks
        person_ids: Dict[int, str] = {}
        for ft in self.current_face_tracks:
            if ft.person_id:
                person_ids[ft.track_id] = ft.person_id

        # Associate faces with bags (using face center as person position)
        assignments = self._associate_faces_bags(self.current_face_tracks, bag_tracks)

        # Behavior analysis
        events = self.behavior.update(bag_tracks, bag_ids, person_ids, assignments)
        if events:
            self.events.emit(events)

        # === Render ===
        self._render(frame, self.current_face_tracks, bag_tracks, bag_ids, assignments)

    def _associate_faces_bags(
        self,
        face_tracks: List[FaceTrack],
        bag_tracks: List
    ) -> Dict[int, int]:
        """Associate bags with nearest face.

        Simple distance-based association using face center.
        """
        assignments: Dict[int, int] = {}  # bag_track_id -> face_track_id

        if not face_tracks or not bag_tracks:
            return assignments

        for bag in bag_tracks:
            bx = (bag.box[0] + bag.box[2]) / 2
            by = (bag.box[1] + bag.box[3]) / 2

            best_face = None
            best_dist = float('inf')

            for ft in face_tracks:
                if ft.person_id is None:
                    continue
                # Face center (use bottom of face as approximate shoulder level)
                fx = (ft.box[0] + ft.box[2]) / 2
                fy = ft.box[3]  # Bottom of face

                dist = np.sqrt((fx - bx) ** 2 + (fy - by) ** 2)
                if dist < best_dist and dist < self.cfg.association.max_link_distance_px * 2:
                    best_dist = dist
                    best_face = ft

            if best_face:
                assignments[bag.track_id] = best_face.track_id

        return assignments

    def _render(
        self,
        frame: np.ndarray,
        face_tracks: List[FaceTrack],
        bag_tracks: List,
        bag_ids: Dict[int, str],
        assignments: Dict[int, int]
    ) -> None:
        """Render face boxes and bag boxes on frame.

        Only draws face bounding boxes (not full body).
        """
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

            # Draw label
            label = ft.person_id if ft.person_id else f"?{ft.track_id}"
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

        # Draw bag boxes
        for bag in bag_tracks:
            x1, y1, x2, y2 = map(int, bag.box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)

            bag_label = bag_ids.get(bag.track_id, f"B{bag.track_id}")

            # Find owner
            owner_track_id = assignments.get(bag.track_id)
            owner_label = "?"
            if owner_track_id:
                for ft in face_tracks:
                    if ft.track_id == owner_track_id and ft.person_id:
                        owner_label = ft.person_id
                        break

            text = f"{bag_label}->{owner_label}"
            cv2.putText(
                frame,
                text,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 165, 255),
                2
            )

        # Show stats
        num_ids = len(self.face_tracker.gallery.get_all_ids())
        stats_text = f"Faces: {len(face_tracks)} | IDs: {num_ids}"
        cv2.putText(frame, stats_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if self.preview:
            self.preview.update(frame)

        if self.render_enabled:
            try:
                cv2.imshow("AntiTerror - Face Tracking", frame)
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
            logger.info(f"Final identity count: {num_ids}")


def parse_args():
    parser = argparse.ArgumentParser(description="Anti-terror video analytics with face tracking")
    parser.add_argument("--source", type=str, default="0", help="Video source (index or path)")
    parser.add_argument("--camera-id", type=str, default="CAM_01", help="Camera identifier")
    parser.add_argument("--conf", type=float, default=None, help="Detection confidence override")
    parser.add_argument("--abandonment-timeout", type=float, default=None, help="Seconds to flag abandoned bag")
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
        pipeline.render_enabled = False  # disable local imshow if serving over HTTP
    pipeline.run()


if __name__ == "__main__":
    main()

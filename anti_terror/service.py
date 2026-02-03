from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Callable, Dict, Optional

import cv2
from loguru import logger

from .config import PipelineConfig, select_device
from .runner import Pipeline


@dataclass(frozen=True)
class CameraConfig:
    camera_id: str
    source: str | int
    db_path: str = "antiterror.db"
    device: str = "cuda"
    preview_port: Optional[int] = None
    render_enabled: bool = False


class MultiCameraService:
    """Run multiple pipelines and expose latest frames/stats for frontend use."""

    def __init__(self):
        self._pipelines: Dict[str, Pipeline] = {}
        self._threads: Dict[str, threading.Thread] = {}
        self._latest_jpeg: Dict[str, bytes] = {}
        self._latest_stats: Dict[str, dict] = {}
        self._lock = threading.Lock()
        self._event_callback: Optional[Callable[[str, list[dict]], None]] = None

    def set_event_callback(self, callback: Callable[[str, list[dict]], None]) -> None:
        """Set a callback to receive events (camera_id, events)."""
        self._event_callback = callback

    def start_camera(self, cfg: CameraConfig) -> None:
        if cfg.camera_id in self._pipelines:
            raise ValueError(f"Camera already running: {cfg.camera_id}")

        pipeline_cfg = PipelineConfig(video_source=cfg.source)
        pipeline_cfg.events.camera_id = cfg.camera_id
        pipeline_cfg.detection.device = select_device(cfg.device)
        pipeline_cfg.embeddings.device = select_device(cfg.device)

        def on_frame(frame, stats):
            # Encode to JPEG once for easy frontend transport.
            try:
                ok, buf = cv2.imencode(".jpg", frame)
                if ok:
                    with self._lock:
                        self._latest_jpeg[cfg.camera_id] = buf.tobytes()
                        self._latest_stats[cfg.camera_id] = stats
            except Exception as e:
                logger.warning(f"JPEG encode failed for {cfg.camera_id}: {e}")

        def on_event(events):
            if self._event_callback:
                try:
                    self._event_callback(cfg.camera_id, events)
                except Exception as e:
                    logger.warning(f"Event callback failed for {cfg.camera_id}: {e}")

        pipeline = Pipeline(
            pipeline_cfg,
            db_path=cfg.db_path,
            frame_callback=on_frame,
            event_callback=on_event,
        )

        if cfg.preview_port:
            from .runner import PreviewServer

            pipeline.preview = PreviewServer(cfg.preview_port)
            pipeline.render_enabled = False
        else:
            pipeline.render_enabled = cfg.render_enabled

        thread = threading.Thread(target=pipeline.run, daemon=True)
        self._pipelines[cfg.camera_id] = pipeline
        self._threads[cfg.camera_id] = thread
        thread.start()

    def stop_camera(self, camera_id: str) -> None:
        pipeline = self._pipelines.get(camera_id)
        if not pipeline:
            return
        pipeline.stop()
        thread = self._threads.get(camera_id)
        if thread:
            thread.join(timeout=5)
        with self._lock:
            self._latest_jpeg.pop(camera_id, None)
            self._latest_stats.pop(camera_id, None)
        self._pipelines.pop(camera_id, None)
        self._threads.pop(camera_id, None)

    def stop_all(self) -> None:
        for camera_id in list(self._pipelines.keys()):
            self.stop_camera(camera_id)

    def get_latest_jpeg(self, camera_id: str) -> Optional[bytes]:
        with self._lock:
            return self._latest_jpeg.get(camera_id)

    def get_latest_stats(self, camera_id: str) -> Optional[dict]:
        with self._lock:
            return self._latest_stats.get(camera_id)

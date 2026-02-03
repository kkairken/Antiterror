from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DetectionConfig:
    # Model selection - YOLO11x is the most accurate
    # Options: "yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt"
    # Or YOLOv8: "yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"
    model_path: str = "yolo11x.pt"  # Best accuracy, use "yolo11s.pt" for speed

    # Detection thresholds
    conf_threshold: float = 0.35  # For persons
    bag_conf_threshold: float = 0.35  # Raised from 0.20 to reduce false positives
    iou_threshold: float = 0.45

    # Input size - larger = better small object detection
    imgsz: int = 640  # Can increase to 1280 for better accuracy (slower)

    # Test-time augmentation - improves accuracy at cost of speed
    augment: bool = True

    device: str = "cuda"  # fallback handled at runtime

    # COCO classes
    classes_person: tuple[int, ...] = (0,)  # person

    # Bag classes - expanded for better coverage
    # 24=backpack, 26=handbag, 28=suitcase
    classes_bag: tuple[int, ...] = (24, 26, 28)


@dataclass
class TrackingConfig:
    # ByteTrack params tuned for real-time webcam
    track_activation_threshold: float = 0.30  # Slightly lower for bags
    lost_track_buffer: int = 90  # 3 seconds at 30fps (was 30) - survives brief occlusions
    minimum_matching_threshold: float = 0.75  # Slightly more permissive
    frame_rate: int = 30
    minimum_consecutive_frames: int = 3  # Filter flicker detections (was 1)


@dataclass
class EmbeddingConfig:
    device: str = "cuda"
    # FaceNet fallback model (used when InsightFace unavailable)
    face_model: str = "vggface2"  # facenet-pytorch pretrained set

    # InsightFace model selection
    # Options: "buffalo_l" (best), "buffalo_m", "buffalo_s", "buffalo_sc"
    # buffalo_l: ArcFace R100 - highest accuracy, recommended for production
    # buffalo_s: ArcFace R34 - faster, good balance
    face_model_name: str = "buffalo_l"

    # Face detection parameters
    face_confidence: float = 0.65  # slightly lower to catch more faces
    face_provider: str = "insightface"  # "insightface" | "facenet"

    # Quality filtering (reduces noise from bad detections)
    min_face_size: int = 50  # minimum face size in pixels
    min_face_quality: float = 0.3  # minimum overall quality score [0-1]

    # ID creation timing (prevent rapid ID creation)
    face_new_id_cooldown_s: float = 5.0  # increased from 3.0 to reduce duplicates
    face_new_id_patience_frames: int = 5  # increased from 3 - wait longer before creating ID
    face_switch_patience_frames: int = 5  # increased from 3 - more stable ID assignment

    # Bag embedding config
    bag_model_name: str = "resnet50"
    bag_embedding_size: int = 2048
    bag_similarity_threshold: float = 0.55  # Lowered from 0.7 for weak ResNet features
    bag_force_match_threshold: float = 0.45  # Force match above this to prevent duplicates
    bag_new_id_patience_frames: int = 5  # Wait before creating new bag ID
    bag_use_color_histogram: bool = True  # Add color features for better matching
    bag_use_shape_features: bool = True  # Add shape features for better matching

    # Face matching thresholds - CRITICAL for reducing duplicates
    # These are optimized for ArcFace/InsightFace embeddings
    face_similarity_threshold: float = 0.55  # lowered from 0.8 - ArcFace scores are typically lower
    face_force_match_margin: float = 0.15  # increased margin for more aggressive matching
    face_create_threshold: float = 0.35  # lowered - prefer reusing existing IDs

    # EMA smoothing parameters
    ema_alpha: float = 0.7  # weight for old embedding (0.7*old + 0.3*new)
    ema_quality_weighted: bool = True  # weight EMA by quality scores


@dataclass
class IdentityConfig:
    """Configuration for identity management and memory bounds."""
    # Memory limits
    max_gallery_size: int = 500  # Maximum identities in memory
    identity_ttl_hours: float = 8.0  # TTL for eviction (hours)

    # Relink after occlusion
    relink_window_s: float = 30.0  # Window for relink after track loss
    relink_threshold: float = 0.45  # Similarity threshold for relink

    # Maintenance intervals (in frames, ~30fps)
    cleanup_interval_frames: int = 900  # ~30 seconds
    db_save_interval_frames: int = 300  # ~10 seconds

    # Track association
    track_merge_threshold: float = 0.50  # Threshold to merge tracks


@dataclass
class AssociationConfig:
    # Distance thresholds
    max_link_distance_px: int = 150  # Increased from 120 for better coverage
    carrying_distance_px: int = 80  # Very close = definitely carrying
    far_distance_px: int = 300  # Far = can see but not carrying

    iou_threshold: float = 0.2
    carrying_iou_threshold: float = 0.1  # Bag overlaps with person body

    # Ownership validation
    time_consistency_frames: int = 10  # Was 8
    transfer_confirmation_frames: int = 30  # ~1 sec for ownership transfer
    minimum_validation_frames: int = 15  # ~0.5 sec to validate ownership
    minimum_validation_confidence: float = 0.6

    # Temporal voting (new)
    history_window_frames: int = 60  # ~2 seconds of history
    ambiguity_margin: float = 0.15  # Score margin between candidates for ambiguity

    # Confidence decay when owner not visible
    confidence_decay_rate: float = 0.05  # Decay per second
    min_confidence_floor: float = 0.1  # Minimum confidence (never fully lose ownership)

    # Scoring weights for temporal voting
    weight_proximity: float = 0.50  # 50% proximity
    weight_temporal: float = 0.30  # 30% temporal consistency
    weight_iou: float = 0.20  # 20% IoU overlap


@dataclass
class BehaviorConfig:
    abandonment_timeout_s: float = 10.0  # Increased from 7.0 for fewer false alarms
    static_iou_threshold: float = 0.7  # Lowered from 0.8 - more tolerant of small movements
    owner_distance_px: int = 250  # Increased from 200
    alert_cooldown_s: float = 30.0  # Don't re-alert same bag for 30 seconds
    stale_track_timeout_s: float = 60.0  # Remove tracks not updated for 60s
    min_confidence_for_abandonment: float = 0.5  # Minimum ownership confidence


@dataclass
class EventConfig:
    camera_id: str = "CAM_01"
    log_dir: Path = Path("logs")
    enable_file_logging: bool = True


@dataclass
class PersistenceConfig:
    save_faces: bool = True
    save_bags: bool = True
    faces_dir: Path = Path("storage/faces")
    bags_dir: Path = Path("storage/bags")


@dataclass
class PipelineConfig:
    video_source: str | int = 0  # default webcam
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    embeddings: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    association: AssociationConfig = field(default_factory=AssociationConfig)
    behavior: BehaviorConfig = field(default_factory=BehaviorConfig)
    events: EventConfig = field(default_factory=EventConfig)
    persistence: PersistenceConfig = field(default_factory=PersistenceConfig)
    identity: IdentityConfig = field(default_factory=IdentityConfig)


def select_device(requested: str) -> str:
    """Pick device string depending on availability.

    Supports:
    - cuda: NVIDIA GPU (Linux/Windows)
    - mps: Apple Silicon GPU (macOS M1/M2/M3/M4)
    - cpu: Fallback for all platforms
    """
    try:
        import torch

        if requested == "cuda" and torch.cuda.is_available():
            return "cuda"
        # Apple Silicon (M1/M2/M3/M4) support via Metal Performance Shaders
        if requested in ("cuda", "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        return "cpu"
    return "cpu"

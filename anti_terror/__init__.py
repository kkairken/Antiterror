"""
Anti-terror video analytics MVP package.

Modules:
- config: tunable parameters.
- video: video capture helpers.
- detection: YOLO-based detector.
- tracking: multi-object tracker wrapper.
- embeddings: face and bag embedding extractors.
- association: logic to bind persons and bags.
- behavior: scenario analysis (abandoned bag).
- events: event formatting/output.
- runner: pipeline orchestration.
- service: multi-camera service and frontend hooks.
"""

__all__ = [
    "config",
    "video",
    "detection",
    "tracking",
    "embeddings",
    "association",
    "behavior",
    "events",
    "runner",
    "service",
]

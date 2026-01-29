"""API module for traffic detection server."""

from .streaming import TrafficAnalysisTrack
from .routes import (
    create_config_router,
    create_events_router,
    create_video_router
)

__all__ = [
    "TrafficAnalysisTrack",
    "create_config_router",
    "create_events_router",
    "create_video_router",
]

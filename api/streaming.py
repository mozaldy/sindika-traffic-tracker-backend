"""WebRTC video streaming with traffic analysis."""

import cv2
import time
import asyncio
import logging
from typing import List, Optional

from aiortc import VideoStreamTrack
from av import VideoFrame

from engine import (
    ObjectDetector, 
    ObjectTracker, 
    StatsManager, 
    TrafficVisualizer,
    MultiLaneSpeedEstimator,
    LicensePlateReader
)
from db import DatabaseManager
from config import ConfigManager
from constants import COCO_CLASSES

logger = logging.getLogger(__name__)


class TrafficAnalysisTrack(VideoStreamTrack):
    """
    WebRTC video track that processes frames through the traffic analysis pipeline.
    
    Reads frames from a video source, runs detection, tracking, speed estimation,
    and license plate reading, then yields annotated frames via WebRTC.
    """
    
    kind = "video"
    
    def __init__(
        self, 
        source_path: str, 
        config_manager: ConfigManager,
        target_classes: Optional[List[str]] = None
    ):
        """
        Initialize the traffic analysis track.
        
        Args:
            source_path: Path to video file or camera ID
            config_manager: Configuration manager instance
            target_classes: List of class names to detect (None = default classes)
        """
        super().__init__()
        self.source_path = source_path
        self.config_manager = config_manager
        
        logger.info(f"Initializing TrafficAnalysisTrack for source: {source_path}")
        
        # Convert class names to IDs
        target_ids = self._resolve_target_classes(target_classes)
        
        # Initialize pipeline components
        self.detector = ObjectDetector(
            confidence_threshold=0.5, 
            target_classes=target_ids
        )
        self.tracker = ObjectTracker()
        self.stats = StatsManager()
        self.visualizer = TrafficVisualizer()
        self.speed_estimator = MultiLaneSpeedEstimator()
        self.plate_reader = LicensePlateReader()
        self.db = DatabaseManager()
        
        # Apply configuration
        lanes = config_manager.get_lanes()
        if lanes:
            self.speed_estimator.set_config(lanes)
        else:
            logger.warning("No lane configuration found")
        
        # Video capture
        self._init_video_capture()
        
        # Timing
        self._frame_interval = 1 / self.fps if self.fps > 0 else 1/30
        self._last_frame_time = 0.0
        self._video_start_time = time.time()
        
        # State
        self._paused = False
        self._last_annotated_frame = None
        
        logger.info(f"TrafficAnalysisTrack ready. FPS: {self.fps}")

    def pause(self) -> None:
        """Pause video processing."""
        self._paused = True
        logger.info("TrafficAnalysisTrack paused")

    def resume(self) -> None:
        """Resume video processing."""
        self._paused = False
        # Adjust start time to account for pause duration?
        # For simplicity, we just resume reading. 
        # Ideally we should shift _video_start_time to avoid jump in timestamps.
        self._video_start_time = time.time() - (self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0)
        logger.info("TrafficAnalysisTrack resumed")

    def _resolve_target_classes(self, target_classes: Optional[List[str]]) -> Optional[List[int]]:
        """Convert class names to COCO class IDs."""
        if not target_classes:
            return None
        
        name_to_id = {v: k for k, v in COCO_CLASSES.items()}
        target_ids = []
        
        for name in target_classes:
            if name in name_to_id:
                target_ids.append(name_to_id[name])
            else:
                logger.warning(f"Unknown class: '{name}'")
        
        return target_ids if target_ids else None

    def _init_video_capture(self) -> None:
        """Initialize video capture from source."""
        self.cap = cv2.VideoCapture(self.source_path)
        
        if not self.cap.isOpened():
            logger.error(f"Failed to open video source: {self.source_path}")
            raise RuntimeError(f"Cannot open video: {self.source_path}")
        
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"Video opened: {self.frame_count} frames @ {self.fps} FPS")

    async def recv(self) -> VideoFrame:
        """
        Receive the next processed video frame.
        
        Called by WebRTC to get each frame for streaming.
        
        Returns:
            Annotated VideoFrame
        """
        # Pace the frame delivery
        pts, time_base = await self.next_timestamp()
        
        if self._paused and self._last_annotated_frame is not None:
            # If paused, return the last frame repeatedly
            # We add a small sleep to avoid tight loop since next_timestamp handles pacing mostly?
            # actually next_timestamp waits for the right time.
            frame = self._last_annotated_frame
        else:
            # Read and process frame
            frame = self._read_frame()
            if frame is None:
                # Create black frame if video ended
                frame = self._create_placeholder_frame()
            else:
                frame = self._process_frame(frame)
                self._last_annotated_frame = frame
        
        # Convert to VideoFrame
        video_frame = VideoFrame.from_ndarray(frame, format="bgr24")
        video_frame.pts = pts
        video_frame.time_base = time_base
        
        return video_frame

    def _read_frame(self) -> Optional[cv2.typing.MatLike]:
        """Read next frame from video source with looping."""
        ret, frame = self.cap.read()
        
        if not ret:
            # Loop video
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self._video_start_time = time.time()
            ret, frame = self.cap.read()
            
            if not ret:
                return None
        
        return frame

    def _create_placeholder_frame(self) -> cv2.typing.MatLike:
        """Create a black placeholder frame."""
        import numpy as np
        return np.zeros((480, 640, 3), dtype=np.uint8)

    def _process_frame(self, frame: cv2.typing.MatLike) -> cv2.typing.MatLike:
        """
        Process a single frame through the analysis pipeline.
        
        Args:
            frame: Raw BGR frame
            
        Returns:
            Annotated BGR frame
        """
        # Calculate video timestamp
        frame_pos = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        timestamp = self._video_start_time + frame_pos
        
        # Detection pipeline
        detections = self.detector.detect(frame)
        detections = self.tracker.update(detections)
        
        # Update statistics and speed estimation
        self.stats.update(detections)
        self.speed_estimator.update(detections, frame.shape, timestamp)
        
        # Log completed crossings
        self._log_completed_crossings(frame, detections, timestamp)
        
        # Annotate frame
        annotated = self.visualizer.annotate(
            frame, detections, self.stats, self.speed_estimator
        )
        
        return annotated

    def _log_completed_crossings(
        self, 
        frame: cv2.typing.MatLike,
        detections,
        timestamp: float
    ) -> None:
        """Log events for vehicles that completed crossing."""
        if detections.tracker_id is None:
            return
        
        for xyxy, class_id, tracker_id in zip(
            detections.xyxy, 
            detections.class_id, 
            detections.tracker_id
        ):
            if tracker_id not in self.speed_estimator.completed_speeds:
                continue
            
            event = self.speed_estimator.completed_speeds[tracker_id]
            if event.get("logged", False):
                continue
            
            # Extract vehicle crop for plate detection
            license_plate, plate_crop = self._detect_license_plate(frame, xyxy)
            
            # Log event
            class_name = COCO_CLASSES.get(class_id, f"class_{class_id}")
            self.db.log_event(
                frame=frame,
                bbox=xyxy.tolist(),
                class_name=class_name,
                speed=event["speed"],
                direction=event["direction"],
                direction_symbol=event.get("direction_symbol"),
                video_source=self.source_path,
                crossing_start=event.get("start_time", 0),
                crossing_end=event.get("end_time", timestamp),
                license_plate=license_plate,
                plate_crop=plate_crop
            )
            
            # Mark as logged
            event["logged"] = True

    def _detect_license_plate(
        self, 
        frame: cv2.typing.MatLike,
        xyxy
    ) -> tuple:
        """Detect and read license plate from vehicle crop."""
        x1, y1, x2, y2 = map(int, xyxy)
        h, w = frame.shape[:2]
        
        # Clamp coordinates
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        if x2 <= x1 or y2 <= y1:
            return None, None
        
        vehicle_crop = frame[y1:y2, x1:x2]
        return self.plate_reader.detect_and_read(vehicle_crop)

    def update_config(self) -> None:
        """Reload configuration from config manager."""
        self.config_manager.reload()
        lanes = self.config_manager.get_lanes()
        self.speed_estimator.set_config(lanes)
        logger.info("Track configuration updated")

    def stop(self) -> None:
        """Release resources."""
        if self.cap:
            self.cap.release()
        logger.info("TrafficAnalysisTrack stopped")

"""WebRTC video streaming with modular traffic analysis pipeline."""

import cv2
import time
import asyncio
import logging
from typing import Dict, List, Optional

import numpy as np
from aiortc import VideoStreamTrack
from av import VideoFrame

from engine import (
    ObjectDetector, 
    ObjectTracker, 
    StatsManager, 
    TrafficVisualizer,
    MultiLaneSpeedEstimator,
    LicensePlateReader,
    VehicleStateManager,
    TurnDetector
)
from db import DatabaseManager
from config import ConfigManager
from constants import COCO_CLASSES

logger = logging.getLogger(__name__)


class TrafficAnalysisTrack(VideoStreamTrack):
    """
    WebRTC video track that processes frames through a modular traffic analysis pipeline.
    
    The pipeline consists of:
    1. Core (always on): Detection → Tracking → State Management
    2. Modules (configurable): Speed, Direction, Plate Detection
    
    Modules can be enabled/disabled via configuration without code changes.
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
        
        # Core pipeline components (always active)
        self.detector = ObjectDetector(
            confidence_threshold=0.5, 
            target_classes=target_ids
        )
        self.tracker = ObjectTracker()
        self.state_manager = VehicleStateManager()
        
        # Analysis modules (conditionally active)
        # Analysis modules (conditionally active)
        self.speed_estimator = MultiLaneSpeedEstimator()
        self.turn_detector = TurnDetector()
        self.plate_reader = LicensePlateReader()
        
        # Support components
        self.stats = StatsManager()
        self.visualizer = TrafficVisualizer()
        self.db = DatabaseManager()
        
        # Apply lane configuration
        lanes = config_manager.get_lanes()
        if lanes:
            self.speed_estimator.set_config(lanes)
        else:
            logger.warning("No lane configuration found")
            
        # Apply zone configuration - filter by type
        zones = config_manager.get_zones()
        if zones:
            # Pass all zones to visualizer for rendering with type-based colors
            self.visualizer.set_zones(zones)
            
            # Set direction zones for turn detector (only first direction zone for now)
            direction_zones = [z for z in zones if z.get("type") == "direction"]
            if direction_zones:
                self.turn_detector.set_zone(direction_zones[0].get("polygon", []))
            
            # Store plate zones for plate capture logic
            self.plate_zones = [z for z in zones if z.get("type") == "plate"]
        else:
            self.plate_zones = []
        
        # Apply plate line configuration
        plate_line = config_manager.get_plate_line()
        if plate_line:
            self.visualizer.set_plate_line(plate_line)
        
        # Video capture
        self._init_video_capture()
        
        # Timing
        self._frame_interval = 1 / self.fps if self.fps > 0 else 1/30
        self._last_frame_time = 0.0
        self._video_start_time = time.time()
        
        # State
        self._paused = False
        self._last_annotated_frame = None
        self._last_positions: Dict[int, tuple] = {}  # Track previous positions for line crossing
        
        # Log active modules
        modules = config_manager.get_modules()
        logger.info(f"TrafficAnalysisTrack ready. FPS: {self.fps}")
        logger.info(f"Active modules: {[k for k, v in modules.items() if v]}")

    def pause(self) -> None:
        """Pause video processing."""
        self._paused = True
        logger.info("TrafficAnalysisTrack paused")

    def resume(self) -> None:
        """Resume video processing."""
        self._paused = False
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
        pts, time_base = await self.next_timestamp()
        
        if self._paused and self._last_annotated_frame is not None:
            frame = self._last_annotated_frame
        else:
            frame = self._read_frame()
            if frame is None:
                frame = self._create_placeholder_frame()
            else:
                frame = self._process_frame(frame)
                self._last_annotated_frame = frame
        
        video_frame = VideoFrame.from_ndarray(frame, format="bgr24")
        video_frame.pts = pts
        video_frame.time_base = time_base
        
        return video_frame

    def _read_frame(self) -> Optional[cv2.typing.MatLike]:
        """Read next frame from video source with looping."""
        ret, frame = self.cap.read()
        
        if not ret:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self._video_start_time = time.time()
            self.state_manager.reset()  # Reset state on video loop
            ret, frame = self.cap.read()
            
            if not ret:
                return None
        
        return frame

    def _create_placeholder_frame(self) -> cv2.typing.MatLike:
        """Create a black placeholder frame."""
        return np.zeros((480, 640, 3), dtype=np.uint8)

    def _process_frame(self, frame: cv2.typing.MatLike) -> cv2.typing.MatLike:
        """
        Process a single frame through the modular analysis pipeline.
        
        Pipeline stages:
        1. Detection (core) - Run object detection
        2. Tracking (core) - Update object tracker
        3. State Management (core) - Update VehicleStateManager
        4. Speed Module (optional) - Calculate speed for detected vehicles
        5. Direction Module (optional) - Calculate movement direction
        6. Plate Module (optional) - Capture license plates based on trigger
        7. Logging - Log completed events to database
        8. Visualization - Annotate frame for display
        
        Args:
            frame: Raw BGR frame
            
        Returns:
            Annotated BGR frame
        """
        try:
            # Calculate video timestamp
            frame_pos = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            timestamp = self._video_start_time + frame_pos
            
            # === CORE PIPELINE (always runs) ===
            detections = self.detector.detect(frame)
            detections = self.tracker.update(detections)
            self.state_manager.update(detections, timestamp)
            self.stats.update(detections)
            
            # Get module configuration
            modules = self.config_manager.get_modules()
            
            # === SPEED MODULE ===
            if modules.get("speed", True):
                self.speed_estimator.update(detections, frame.shape, timestamp)
                self._sync_speed_to_state()
                
            # === TURN MODULE ===
            if modules.get("turn", False):
                self.turn_detector.update(detections, frame.shape, timestamp)
                self._sync_turn_to_state()
            
            # === PLATE MODULE ===
            if modules.get("plate", False):
                trigger = self.config_manager.get_plate_trigger()
                if trigger == "on_line":
                    self._check_plate_line_crossings(detections, frame)
                else:
                    self._process_plate_captures(frame)
            
            # === LOGGING ===
            self._log_completed_vehicles(frame, timestamp)
            
            # === VISUALIZATION ===
            annotated = self.visualizer.annotate(
                frame, detections, self.stats, self.speed_estimator, self.turn_detector
            )
            
            return annotated
            
        except Exception as e:
            logger.error(f"Error processing frame: {e}", exc_info=True)
            # Return original frame on error to keep stream alive
            return frame

    def _sync_speed_to_state(self) -> None:
        """Sync speed estimator results to VehicleStateManager."""
        for track_id, speed_data in self.speed_estimator.completed_speeds.items():
            vehicle = self.state_manager.get_vehicle(track_id)
            if vehicle and vehicle.speed_kmh is None:
                vehicle.speed_kmh = speed_data.get("speed")
                vehicle.crossing_start = speed_data.get("start_time")
                vehicle.crossing_end = speed_data.get("end_time")
                vehicle.lane_name = speed_data.get("lane_name")
                
                # Default to Straight/Forward if valid speed but no turn detected yet
                if not vehicle.direction_symbol:
                    vehicle.direction_symbol = "⭡"
                    
                self.state_manager.mark_completed(track_id)

    def _sync_turn_to_state(self) -> None:
        """Sync turn detector results to VehicleStateManager."""
        for vehicle in self.state_manager.vehicles.values():
            # Check if already completed (by speed or turn)
            if vehicle.track_id in self.state_manager.completed_vehicles:
                continue
                
            turn_result = self.turn_detector.get_turn_for_vehicle(vehicle.track_id)
            if turn_result:
                # Update vehicle state with turn info
                vehicle.direction_symbol = turn_result.turn_symbol
                vehicle.direction_deg = turn_result.direction_deg


    def _check_plate_line_crossings(self, detections, frame: cv2.typing.MatLike) -> None:
        """Check if any vehicle crossed the plate capture line and capture if so."""
        plate_line = self.config_manager.get_plate_line()
        if not plate_line or len(plate_line) != 4:
            return
        
        if detections.tracker_id is None:
            return
        
        h, w = frame.shape[:2]
        lx1, ly1, lx2, ly2 = plate_line[0]*w, plate_line[1]*h, plate_line[2]*w, plate_line[3]*h
        
        for tracker_id, box in zip(detections.tracker_id, detections.xyxy):
            tracker_id = int(tracker_id)
            vehicle = self.state_manager.get_vehicle(tracker_id)
            if not vehicle or vehicle.plate_captured:
                continue
            
            # Get current center position
            curr_pos = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
            prev_pos = self._last_positions.get(tracker_id)
            self._last_positions[tracker_id] = curr_pos
            
            if prev_pos is None:
                continue
            
            # Check line crossing using CCW method
            if self._segments_intersect(prev_pos, curr_pos, (lx1, ly1), (lx2, ly2)):
                self._capture_vehicle_and_plate(frame, vehicle)
                logger.info(f"Plate line capture triggered for vehicle {tracker_id}")

    def _log_completed_vehicles(self, frame: cv2.typing.MatLike, timestamp: float) -> None:
        """Log completed vehicle events to database."""
        for vehicle in self.state_manager.get_completed_unlogged():
            
            # Filter Invalid: No Speed AND No Turn -> Skip
            if (vehicle.speed_kmh or 0) <= 0 and not vehicle.direction_symbol:
                # Mark as logged so we don't retry forever? 
                # actually get_completed_unlogged returns completed vehicles.
                # if we skip, they remain unlogged=False? 
                # state_manager.mark_logged(track_id) should be called to clear them.
                vehicle.is_logged = True 
                continue

            # Use pre-captured images if available (preferred)
            image_path = vehicle.vehicle_image_path
            plate_image_path = vehicle.plate_image_path
            plate_crop = None
        """Check if any vehicle crossed the plate capture line and capture if so."""
        plate_line = self.config_manager.get_plate_line()
        if not plate_line or len(plate_line) != 4:
            return
        
        if detections.tracker_id is None:
            return
        
        h, w = frame.shape[:2]
        lx1, ly1, lx2, ly2 = plate_line[0]*w, plate_line[1]*h, plate_line[2]*w, plate_line[3]*h
        
        for tracker_id, box in zip(detections.tracker_id, detections.xyxy):
            tracker_id = int(tracker_id)
            vehicle = self.state_manager.get_vehicle(tracker_id)
            if not vehicle or vehicle.plate_captured:
                continue
            
            # Get current center position
            curr_pos = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
            prev_pos = self._last_positions.get(tracker_id)
            self._last_positions[tracker_id] = curr_pos
            
            if prev_pos is None:
                continue
            
            # Check line crossing using CCW method
            if self._segments_intersect(prev_pos, curr_pos, (lx1, ly1), (lx2, ly2)):
                self._capture_vehicle_and_plate(frame, vehicle)
                logger.info(f"Plate line capture triggered for vehicle {tracker_id}")

    def _segments_intersect(self, A: tuple, B: tuple, C: tuple, D: tuple) -> bool:
        """Check if segment AB intersects segment CD."""
        def ccw(P, Q, R):
            return (R[1] - P[1]) * (Q[0] - P[0]) > (Q[1] - P[1]) * (R[0] - P[0])
        return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

    def _capture_vehicle_and_plate(self, frame: cv2.typing.MatLike, vehicle) -> None:
        """Capture vehicle and plate images immediately and save to database."""
        if vehicle.current_bbox is None:
            return
        
        x1, y1, x2, y2 = map(int, vehicle.current_bbox)
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        if x2 <= x1 or y2 <= y1:
            return
        
        # Capture and detect plate
        vehicle_crop = frame[y1:y2, x1:x2]
        plate_text, plate_crop = self.plate_reader.detect_and_read(vehicle_crop)
        
        vehicle.plate_text = plate_text
        vehicle.plate_captured = True
        
        # Save images immediately
        image_path, plate_image_path = self.db.save_captures(
            frame=frame,
            bbox=vehicle.current_bbox,
            class_name=vehicle.class_name,
            speed=vehicle.speed_kmh or 0,
            plate_crop=plate_crop
        )
        
        vehicle.vehicle_image_path = image_path
        vehicle.plate_image_path = plate_image_path
        
        # Mark as completed so it gets picked up for logging once stale
        self.state_manager.mark_completed(vehicle.track_id)

    def _process_plate_captures(self, frame: cv2.typing.MatLike) -> None:
        """Process plate captures based on trigger configuration (legacy/fallback)."""
        trigger = self.config_manager.get_plate_trigger()
        threshold = self.config_manager.get_speed_threshold()
        
        for vehicle in self.state_manager.vehicles.values():
            if self.plate_reader.should_capture(vehicle, trigger, threshold):
                self._capture_plate(frame, vehicle)

    def _capture_plate(self, frame: cv2.typing.MatLike, vehicle) -> None:
        """Capture and store license plate for a vehicle."""
        if vehicle.current_bbox is None:
            return
        
        x1, y1, x2, y2 = map(int, vehicle.current_bbox)
        h, w = frame.shape[:2]
        
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        if x2 <= x1 or y2 <= y1:
            return
        
        vehicle_crop = frame[y1:y2, x1:x2]
        plate_text, plate_crop = self.plate_reader.detect_and_read(vehicle_crop)
        
        vehicle.plate_text = plate_text
        vehicle.plate_captured = True
        
        # Save images immediately
        image_path, plate_image_path = self.db.save_captures(
            frame=frame,
            bbox=vehicle.current_bbox,
            class_name=vehicle.class_name,
            speed=vehicle.speed_kmh or 0,
            plate_crop=plate_crop
        )
        
        vehicle.vehicle_image_path = image_path
        vehicle.plate_image_path = plate_image_path
        
        # Mark as completed (optional, or let it complete naturally on exit)
        # If trigger is 'always' or 'speed', we might want to let it continue tracking
        # But we captured the 'event', so we mark completed to ensure it gets logged even if it stays in frame?
        # Actually, for 'always', we might want to capture once.
        # Let's mark as completed to ensure logging happens eventually.
        self.state_manager.mark_completed(vehicle.track_id)
        
        if plate_crop is not None:
            logger.debug(f"Plate captured for vehicle {vehicle.track_id}")

    def _log_completed_vehicles(self, frame: cv2.typing.MatLike, timestamp: float) -> None:
        """Log completed vehicle events to database."""
        for vehicle in self.state_manager.get_completed_unlogged():
            # Use pre-captured images if available (preferred)
            image_path = vehicle.vehicle_image_path
            plate_image_path = vehicle.plate_image_path
            plate_crop = None
            
            # Fallback: if no pre-captured plate but we have a bbox (rare for stale vehicles), try to grab it
            # This is mostly legacy behavior for non-trigger based logging
            if not plate_image_path and vehicle.current_bbox is not None:
                 # ... existing fallback logic or just skip ...
                 pass

            self.db.log_event(
                frame=frame,
                bbox=vehicle.current_bbox,
                class_name=vehicle.class_name,
                speed=vehicle.speed_kmh or 0,
                direction=vehicle.direction_deg or 0,
                direction_symbol=vehicle.direction_symbol,
                video_source=self.source_path,
                crossing_start=vehicle.crossing_start or vehicle.first_seen,
                crossing_end=vehicle.crossing_end or timestamp,
                license_plate=vehicle.plate_text,
                plate_crop=plate_crop,
                image_path=image_path,
                plate_image_path=plate_image_path
            )
            
            self.state_manager.mark_logged(vehicle.track_id)
            speed_str = f"{vehicle.speed_kmh:.1f}" if vehicle.speed_kmh else "0"
            logger.info(f"Logged vehicle {vehicle.track_id}: {vehicle.class_name}, {speed_str} km/h")

    def update_config(self) -> None:
        """Reload configuration from config manager."""
        self.config_manager.reload()
        lanes = self.config_manager.get_lanes()
        self.speed_estimator.set_config(lanes)
        
        zones = self.config_manager.get_zones()
        if zones:
            # Pass all zones to visualizer
            self.visualizer.set_zones(zones)
            
            # Set direction zones for turn detector
            direction_zones = [z for z in zones if z.get("type") == "direction"]
            if direction_zones:
                self.turn_detector.set_zone(direction_zones[0].get("polygon", []))
            else:
                self.turn_detector.set_zone([])
            
            # Update plate zones
            self.plate_zones = [z for z in zones if z.get("type") == "plate"]
        else:
            self.turn_detector.set_zone([])
            self.visualizer.set_zones([])
            self.plate_zones = []
        
        # Update plate line
        plate_line = self.config_manager.get_plate_line()
        self.visualizer.set_plate_line(plate_line)
            
        logger.info("Track configuration updated")

    def stop(self) -> None:
        """Release resources."""
        if self.cap:
            self.cap.release()
        self.state_manager.reset()
        logger.info("TrafficAnalysisTrack stopped")

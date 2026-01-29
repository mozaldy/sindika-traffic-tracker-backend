"""Turn detection module using polygon zones."""

import cv2
import math
import logging
import numpy as np
from typing import Dict, Set, List, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class TurnResult:
    """Result of a completed turn detection."""
    track_id: int
    entry_pos: Tuple[float, float]
    exit_pos: Tuple[float, float]
    direction_deg: float  # Angle in degrees
    turn_type: str  # "forward", "left", "right", "uturn"
    turn_symbol: str  # Arrow symbol
    timestamp: float


class TurnDetector:
    """
    Polygon zone-based turn detector with angle-based driver perspective.
    
    Tracks vehicles passing through a polygon zone and determines
    turning behavior based on movement angle from entry to exit.
    
    Direction is calculated from vehicle's driver perspective:
    - Forward: Moving towards top of frame
    - Left: Turning left from driver's POV
    - Right: Turning right from driver's POV
    """
    
    # Turn symbols based on driver perspective
    TURN_SYMBOLS = {
        "forward": "⭡",
        "left": "↰",
        "right": "↱",
        "uturn": "⭣",
        "unknown": "?"
    }
    
    def __init__(self):
        self.zone_polygon: Optional[np.ndarray] = None
        self.entry_positions: Dict[int, Tuple[float, float]] = {}  # track_id -> entry position
        self.objects_in_zone: Set[int] = set()
        self.last_positions: Dict[int, Tuple[float, float]] = {}
        self.completed_turns: Dict[int, TurnResult] = {}
        self.active_trails: Dict[int, List[Tuple[float, float]]] = {}
        
    def set_zone(self, polygon_data) -> None:
        """
        Set polygon zone from normalized points.
        
        Args:
            polygon_data: Either flat array [x1,y1,x2,y2,...] or nested [[x1,y1], [x2,y2], ...]
        """
        if not polygon_data:
            logger.warning("Empty zone polygon data")
            self.zone_polygon = None
            return
            
        # Handle flat array format [x1,y1,x2,y2,...]
        if isinstance(polygon_data, list) and len(polygon_data) > 0:
            if isinstance(polygon_data[0], (int, float)):
                # Flat array - convert to points
                points = []
                for i in range(0, len(polygon_data), 2):
                    if i + 1 < len(polygon_data):
                        points.append([polygon_data[i], polygon_data[i+1]])
                polygon_data = points
        
        if polygon_data and len(polygon_data) >= 3:
            self.zone_polygon = np.array(polygon_data, dtype=np.float32)
            logger.info(f"Turn zone set with {len(polygon_data)} vertices")
        else:
            logger.warning("Invalid zone points - need at least 3 vertices")
            self.zone_polygon = None
    
    def reset(self) -> None:
        """Reset all tracking state."""
        self.entry_positions.clear()
        self.objects_in_zone.clear()
        self.last_positions.clear()
        self.completed_turns.clear()
        self.active_trails.clear()
    
    def update(self, detections, frame_shape: Tuple[int, ...], timestamp: float) -> None:
        """
        Update turn tracking with new detections.
        
        Args:
            detections: Supervision detections with tracker_id
            frame_shape: (height, width, channels)
            timestamp: Current timestamp
        """
        if self.zone_polygon is None or detections.tracker_id is None:
            return
        
        h, w = frame_shape[:2]
        zone_pixel = (self.zone_polygon * np.array([w, h], dtype=np.float32)).astype(np.int32)
        current_ids = set()
        
        for tracker_id, box in zip(detections.tracker_id, detections.xyxy):
            current_ids.add(tracker_id)
            
            # Calculate center position
            curr_pos = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
            
            # Update trail
            if tracker_id not in self.active_trails:
                self.active_trails[tracker_id] = []
            self.active_trails[tracker_id].append(curr_pos)
            
            # Check if inside zone
            is_inside = cv2.pointPolygonTest(zone_pixel, curr_pos, False) >= 0
            was_inside = tracker_id in self.objects_in_zone
            
            # Zone entry - record entry position
            if is_inside and not was_inside:
                self.objects_in_zone.add(tracker_id)
                self.entry_positions[tracker_id] = curr_pos
                logger.debug(f"Vehicle {tracker_id} entered zone at {curr_pos}")
            
            # Zone exit - calculate direction from entry to exit
            elif not is_inside and was_inside:
                self.objects_in_zone.discard(tracker_id)
                
                if tracker_id in self.entry_positions:
                    entry_pos = self.entry_positions[tracker_id]
                    turn_type, direction_deg = self._calculate_direction(entry_pos, curr_pos)
                    
                    self.completed_turns[tracker_id] = TurnResult(
                        track_id=tracker_id,
                        entry_pos=entry_pos,
                        exit_pos=curr_pos,
                        direction_deg=direction_deg,
                        turn_type=turn_type,
                        turn_symbol=self.TURN_SYMBOLS.get(turn_type, "?"),
                        timestamp=timestamp
                    )
                    
                    logger.info(
                        f"Vehicle {tracker_id}: {turn_type.upper()} ({direction_deg:.0f}°) {self.TURN_SYMBOLS.get(turn_type)}"
                    )
            
            self.last_positions[tracker_id] = curr_pos
        
        # Cleanup stale entries
        stale_ids = set(self.last_positions.keys()) - current_ids
        for stale_id in stale_ids:
            self._cleanup_vehicle(stale_id)
    
    def _cleanup_vehicle(self, track_id: int) -> None:
        """Remove tracking data for a vehicle."""
        self.last_positions.pop(track_id, None)
        self.entry_positions.pop(track_id, None)
        self.active_trails.pop(track_id, None)
        self.objects_in_zone.discard(track_id)
    
    def _calculate_direction(
        self, 
        entry_pos: Tuple[float, float], 
        exit_pos: Tuple[float, float]
    ) -> Tuple[str, float]:
        """
        Calculate turn type from entry to exit position using angle-based driver perspective.
        
        In image coordinates: 
        - Y increases downward
        - A vehicle moving "forward" (away from camera) moves toward smaller Y values
        
        From driver perspective looking at the road:
        - Forward (⭡): Moving up in frame (angle ~270° / -90°)
        - Left (↰): Turning left from driver POV
        - Right (↱): Turning right from driver POV  
        - U-turn (⭣): Turning back
        
        Args:
            entry_pos: Entry position (x, y)
            exit_pos: Exit position (x, y)
            
        Returns:
            Tuple of (turn_type, direction_degrees)
        """
        dx = exit_pos[0] - entry_pos[0]
        dy = exit_pos[1] - entry_pos[1]
        
        # Calculate angle in standard math coordinates (0° = right, CCW positive)
        angle = math.degrees(math.atan2(dy, dx)) % 360
        
        # Convert to driver perspective where 0° = forward (up in image)
        # In image: up is -90°, so rotate by 90° to make up = 0°
        driver_angle = (angle + 90) % 360
        
        # Determine turn type based on driver perspective angle
        # 0° = forward, 90° = right, 180° = back, 270° = left
        if 315 <= driver_angle or driver_angle < 45:
            turn_type = "forward"
        elif 45 <= driver_angle < 135:
            turn_type = "right"
        elif 135 <= driver_angle < 225:
            turn_type = "uturn"
        elif 225 <= driver_angle < 315:
            turn_type = "left"
        else:
            turn_type = "unknown"
        
        return turn_type, angle

    
    def get_turn_for_vehicle(self, track_id: int) -> Optional[TurnResult]:
        """Get turn result for a specific vehicle."""
        return self.completed_turns.get(track_id)
    
    def is_vehicle_in_zone(self, track_id: int) -> bool:
        """Check if vehicle is currently in the zone."""
        return track_id in self.objects_in_zone
    
    def get_zone_polygon_pixels(self, frame_shape: Tuple[int, ...]) -> Optional[np.ndarray]:
        """Get zone polygon in pixel coordinates for visualization."""
        if self.zone_polygon is None:
            return None
        h, w = frame_shape[:2]
        return (self.zone_polygon * np.array([w, h], dtype=np.float32)).astype(np.int32)

"""Multi-lane speed estimation using two-line crossing method."""

import math
import logging
from typing import Dict, List, Set, Optional, Any
from dataclasses import dataclass

import numpy as np
import supervision as sv

logger = logging.getLogger(__name__)


@dataclass
class CrossingEvent:
    """Represents a vehicle crossing a line."""
    lane_idx: int
    timestamp: float
    position: tuple
    entry_line: str  # 'a' or 'b'
    exit_line: str   # 'b' or 'a'


@dataclass
class SpeedResult:
    """Result of a completed speed measurement."""
    speed: float
    direction: float
    direction_symbol: str
    lane_name: str
    timestamp: float
    start_time: float
    end_time: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "speed": self.speed,
            "direction": self.direction,
            "direction_symbol": self.direction_symbol,
            "lane_name": self.lane_name,
            "timestamp": self.timestamp,
            "start_time": self.start_time,
            "end_time": self.end_time
        }


class MultiLaneSpeedEstimator:
    """
    Multi-lane speed estimator using two-line crossing method.
    
    Each lane has an entry line (A) and exit line (B) with a known
    distance between them. Speed is calculated when a vehicle crosses
    both lines: speed = distance / time.
    
    Supports bidirectional detection (A→B and B→A).
    """
    
    # Direction symbols for cardinal directions
    DIRECTION_SYMBOLS = {
        'north': ('⇧', '⇩'),  # (forward, reverse)
        'south': ('⇩', '⇧'),
        'east': ('⇨', '⇦'),
        'west': ('⇦', '⇨'),
    }
    DEFAULT_SYMBOL = ('→', '←')
    
    def __init__(self):
        """Initialize the speed estimator."""
        self.lanes: List[Dict] = []
        self.last_positions: Dict[int, tuple] = {}
        self.active_trails: Dict[int, List[tuple]] = {}
        self.line_crossings: Dict[int, CrossingEvent] = {}
        self.completed_speeds: Dict[int, Dict] = {}
        self.objects_in_lanes: Dict[int, int] = {}
        
        logger.debug("MultiLaneSpeedEstimator initialized.")

    def set_config(self, lanes: List[Dict]) -> None:
        """
        Set lane configurations.
        
        Args:
            lanes: List of lane configs, each with:
                - name: str
                - line_a: [x1, y1, x2, y2] (normalized 0-1)
                - line_b: [x1, y1, x2, y2] (normalized 0-1)
                - distance: float (meters between lines)
        """
        self.lanes = lanes
        logger.info(f"Configured {len(lanes)} lanes for speed detection")
        for i, lane in enumerate(lanes):
            logger.info(f"  Lane {i}: {lane.get('name', f'Lane {i+1}')} - {lane.get('distance', 5)}m")

    def update(self, detections: sv.Detections, frame_shape: tuple, timestamp: float) -> None:
        """
        Update speed estimation with new detections.
        
        Args:
            detections: Current frame detections with tracker IDs
            frame_shape: Shape of the frame (height, width, channels)
            timestamp: Current frame timestamp in seconds
        """
        if not self.lanes or detections.tracker_id is None:
            return
            
        current_ids = set()
        
        for tracker_id, box in zip(detections.tracker_id, detections.xyxy):
            current_ids.add(tracker_id)
            curr_pos = self._get_center(box)
            
            # Track trail
            self._update_trail(tracker_id, curr_pos)
            
            # Skip if no previous position
            if tracker_id not in self.last_positions:
                self.last_positions[tracker_id] = curr_pos
                continue
            
            prev_pos = self.last_positions[tracker_id]
            
            # Check lane crossings
            self._check_lane_crossings(tracker_id, prev_pos, curr_pos, frame_shape, timestamp)
            
            self.last_positions[tracker_id] = curr_pos
        
        # Cleanup old tracked objects
        self._cleanup_stale_objects(current_ids)

    def _get_center(self, box: np.ndarray) -> tuple:
        """Get center point of bounding box."""
        return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)

    def _update_trail(self, tracker_id: int, position: tuple) -> None:
        """Update the trail for an object."""
        if tracker_id not in self.active_trails:
            self.active_trails[tracker_id] = []
        self.active_trails[tracker_id].append(position)

    def _check_lane_crossings(
        self, 
        tracker_id: int, 
        prev_pos: tuple, 
        curr_pos: tuple, 
        frame_shape: tuple, 
        timestamp: float
    ) -> None:
        """Check if object crossed any lane lines."""
        for lane_idx, lane in enumerate(self.lanes):
            line_a = lane.get('line_a')
            line_b = lane.get('line_b')
            distance = lane.get('distance', 5.0)
            lane_name = lane.get('name', f'Lane {lane_idx + 1}')
            
            # Check entry (if not already tracking this object)
            if tracker_id not in self.line_crossings:
                if self._check_line_crossing(prev_pos, curr_pos, line_a, frame_shape):
                    self._record_entry(tracker_id, lane_idx, timestamp, curr_pos, 'a', 'b')
                    logger.debug(f"Object {tracker_id} crossed Line A of {lane_name} (forward)")
                elif self._check_line_crossing(prev_pos, curr_pos, line_b, frame_shape):
                    self._record_entry(tracker_id, lane_idx, timestamp, curr_pos, 'b', 'a')
                    logger.debug(f"Object {tracker_id} crossed Line B of {lane_name} (reverse)")
            
            # Check exit
            elif tracker_id in self.line_crossings:
                crossing = self.line_crossings[tracker_id]
                if crossing.lane_idx == lane_idx:
                    exit_line = line_b if crossing.exit_line == 'b' else line_a
                    
                    if self._check_line_crossing(prev_pos, curr_pos, exit_line, frame_shape):
                        self._complete_crossing(
                            tracker_id, crossing, curr_pos, timestamp,
                            distance, lane_name
                        )

    def _record_entry(
        self, 
        tracker_id: int, 
        lane_idx: int, 
        timestamp: float, 
        position: tuple,
        entry_line: str,
        exit_line: str
    ) -> None:
        """Record when an object enters a lane."""
        self.line_crossings[tracker_id] = CrossingEvent(
            lane_idx=lane_idx,
            timestamp=timestamp,
            position=position,
            entry_line=entry_line,
            exit_line=exit_line
        )
        self.objects_in_lanes[tracker_id] = lane_idx

    def _complete_crossing(
        self,
        tracker_id: int,
        crossing: CrossingEvent,
        exit_position: tuple,
        timestamp: float,
        distance: float,
        lane_name: str
    ) -> None:
        """Complete a crossing and calculate speed."""
        duration = timestamp - crossing.timestamp
        
        if duration > 0.1:  # Minimum 0.1 second crossing
            speed_kmh = (distance / duration) * 3.6
            is_reverse = crossing.entry_line == 'b'
            
            # Calculate direction angle
            dx = exit_position[0] - crossing.position[0]
            dy = exit_position[1] - crossing.position[1]
            angle = math.degrees(math.atan2(dy, dx)) % 360
            
            display_name = f"{lane_name} (Rev)" if is_reverse else lane_name
            
            result = SpeedResult(
                speed=speed_kmh,
                direction=angle,
                direction_symbol=self._get_direction_symbol(lane_name, is_reverse),
                lane_name=display_name,
                timestamp=timestamp,
                start_time=crossing.timestamp,
                end_time=timestamp
            )
            
            self.completed_speeds[tracker_id] = result.to_dict()
            logger.info(f"Object {tracker_id} completed {display_name}: {speed_kmh:.1f} km/h")
        
        # Clear tracking
        del self.line_crossings[tracker_id]
        self.objects_in_lanes.pop(tracker_id, None)

    def _check_line_crossing(
        self, 
        prev_pos: tuple, 
        curr_pos: tuple, 
        line: List[float], 
        frame_shape: tuple
    ) -> bool:
        """Check if movement crossed a line."""
        if not line or len(line) != 4:
            return False
        
        h, w = frame_shape[:2]
        lx1, ly1, lx2, ly2 = line[0]*w, line[1]*h, line[2]*w, line[3]*h
        
        return self._segments_intersect(prev_pos, curr_pos, (lx1, ly1), (lx2, ly2))

    def _segments_intersect(self, A: tuple, B: tuple, C: tuple, D: tuple) -> bool:
        """Check if segment AB intersects segment CD."""
        def ccw(P, Q, R):
            return (R[1] - P[1]) * (Q[0] - P[0]) > (Q[1] - P[1]) * (R[0] - P[0])
        
        return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

    def _get_direction_symbol(self, lane_name: str, is_reverse: bool = False) -> str:
        """Get direction symbol based on lane name."""
        name_lower = lane_name.lower()
        
        for direction, symbols in self.DIRECTION_SYMBOLS.items():
            if direction in name_lower:
                return symbols[1] if is_reverse else symbols[0]
        
        return self.DEFAULT_SYMBOL[1] if is_reverse else self.DEFAULT_SYMBOL[0]

    def _cleanup_stale_objects(self, current_ids: Set[int]) -> None:
        """Remove tracking data for objects no longer visible."""
        stale_ids = set(self.last_positions.keys()) - current_ids
        
        for tid in stale_ids:
            self.last_positions.pop(tid, None)
            self.line_crossings.pop(tid, None)
            self.objects_in_lanes.pop(tid, None)
            self.active_trails.pop(tid, None)

    def reset(self) -> None:
        """Reset all tracking state."""
        self.last_positions.clear()
        self.active_trails.clear()
        self.line_crossings.clear()
        self.completed_speeds.clear()
        self.objects_in_lanes.clear()

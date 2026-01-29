"""Turn detection module using polygon zones."""

import cv2
import logging
import numpy as np
from typing import Dict, Set, List, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class TurnResult:
    """Result of a completed turn detection."""
    track_id: int
    entry_edge: int
    exit_edge: int
    turn_type: str  # "left", "right", "straight"
    turn_symbol: str  # Arrow symbol
    timestamp: float


class TurnDetector:
    """
    Polygon zone-based turn detector.
    
    Tracks vehicles passing through a polygon zone and determines
    turning behavior based on entry and exit edges.
    
    Turn is calculated relative to the vehicle's direction of travel:
    - Straight: Exits through opposite edge
    - Left: Exits through clockwise edge (from vehicle's POV)
    - Right: Exits through counter-clockwise edge (from vehicle's POV)
    """
    
    # Turn symbols based on entry→exit movement
    TURN_SYMBOLS = {
        "straight": "↕",
        "left": "↰",
        "right": "↱",
        "unknown": "?"
    }
    
    def __init__(self):
        self.zone_polygon: Optional[np.ndarray] = None
        self.entry_edges: Dict[int, int] = {}  # track_id -> entry edge index
        self.objects_in_zone: Set[int] = set()
        self.last_positions: Dict[int, Tuple[float, float]] = {}
        self.completed_turns: Dict[int, TurnResult] = {}
        self.active_trails: Dict[int, List[Tuple[float, float]]] = {}
        
    def set_zone(self, points: List[List[float]]) -> None:
        """
        Set polygon zone from normalized points.
        
        Args:
            points: List of [x, y] normalized coordinates (0-1)
        """
        if points and len(points) >= 3:
            self.zone_polygon = np.array(points, dtype=np.float32)
            logger.info(f"Turn zone set with {len(points)} vertices")
        else:
            logger.warning("Invalid zone points - need at least 3 vertices")
            self.zone_polygon = None
    
    def reset(self) -> None:
        """Reset all tracking state."""
        self.entry_edges.clear()
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
            
            # Zone entry
            if is_inside and not was_inside:
                self.objects_in_zone.add(tracker_id)
                
                # Detect entry edge
                if tracker_id in self.last_positions:
                    edge = self._get_crossed_edge(
                        self.last_positions[tracker_id], 
                        curr_pos, 
                        frame_shape
                    )
                    if edge is not None:
                        self.entry_edges[tracker_id] = edge
                        logger.debug(f"Vehicle {tracker_id} entered zone via edge {edge + 1}")
            
            # Zone exit
            elif not is_inside and was_inside:
                self.objects_in_zone.discard(tracker_id)
                
                # Detect exit edge and calculate turn
                if tracker_id in self.entry_edges and tracker_id in self.last_positions:
                    exit_edge = self._get_crossed_edge(
                        self.last_positions[tracker_id],
                        curr_pos,
                        frame_shape
                    )
                    
                    if exit_edge is not None:
                        entry_edge = self.entry_edges[tracker_id]
                        num_edges = len(self.zone_polygon)
                        turn_type = self._calculate_turn(entry_edge, exit_edge, num_edges)
                        
                        self.completed_turns[tracker_id] = TurnResult(
                            track_id=tracker_id,
                            entry_edge=entry_edge,
                            exit_edge=exit_edge,
                            turn_type=turn_type,
                            turn_symbol=self.TURN_SYMBOLS.get(turn_type, "?"),
                            timestamp=timestamp
                        )
                        
                        logger.info(
                            f"Vehicle {tracker_id}: Edge {entry_edge + 1} → Edge {exit_edge + 1} = {turn_type.upper()}"
                        )
            
            self.last_positions[tracker_id] = curr_pos
        
        # Cleanup stale entries
        stale_ids = set(self.last_positions.keys()) - current_ids
        for stale_id in stale_ids:
            self._cleanup_vehicle(stale_id)
    
    def _cleanup_vehicle(self, track_id: int) -> None:
        """Remove tracking data for a vehicle."""
        self.last_positions.pop(track_id, None)
        self.entry_edges.pop(track_id, None)
        self.active_trails.pop(track_id, None)
        self.objects_in_zone.discard(track_id)
    
    def _ccw(self, A: Tuple, B: Tuple, C: Tuple) -> bool:
        """Check if points are counter-clockwise."""
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
    
    def _segments_intersect(self, A: Tuple, B: Tuple, C: Tuple, D: Tuple) -> bool:
        """Check if line segment AB intersects with segment CD."""
        return self._ccw(A, C, D) != self._ccw(B, C, D) and self._ccw(A, B, C) != self._ccw(A, B, D)
    
    def _get_crossed_edge(
        self, 
        prev_pos: Tuple[float, float], 
        curr_pos: Tuple[float, float], 
        frame_shape: Tuple[int, ...]
    ) -> Optional[int]:
        """
        Detect which polygon edge was crossed.
        
        Args:
            prev_pos: Previous position (pixels)
            curr_pos: Current position (pixels)
            frame_shape: Frame dimensions
            
        Returns:
            Edge index (0-based) or None if no edge crossed
        """
        if self.zone_polygon is None:
            return None
        
        h, w = frame_shape[:2]
        poly_pixels = (self.zone_polygon * np.array([w, h], dtype=np.float32)).astype(np.int32)
        
        for i in range(len(poly_pixels)):
            p1 = tuple(poly_pixels[i])
            p2 = tuple(poly_pixels[(i + 1) % len(poly_pixels)])
            
            if self._segments_intersect(prev_pos, curr_pos, p1, p2):
                return i
        
        return None
    
    def _calculate_turn(self, entry_edge: int, exit_edge: int, num_edges: int) -> str:
        """
        Calculate turn type relative to vehicle's direction.
        
        For a 4-sided polygon:
        - Entry 0 → Exit 2 (opposite, diff=2) = Straight
        - Entry 0 → Exit 1 (diff=1, < opposite) = Left
        - Entry 0 → Exit 3 (diff=3, > opposite) = Right
        
        Args:
            entry_edge: Entry edge index (0-based)
            exit_edge: Exit edge index (0-based)
            num_edges: Total number of polygon edges
            
        Returns:
            "straight", "left", or "right"
        """
        if entry_edge == exit_edge:
            return "unknown"  # Same edge - shouldn't happen normally
        
        # Calculate clockwise distance from entry to exit
        diff = (exit_edge - entry_edge) % num_edges
        opposite = num_edges // 2
        
        if diff == opposite:
            return "straight"
        elif diff < opposite:
            return "left"  # Clockwise from vehicle's POV
        else:
            return "right"  # Counter-clockwise from vehicle's POV
    
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

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
            
            # Zone entry - record entry position AND edge
            if is_inside and not was_inside:
                self.objects_in_zone.add(tracker_id)
                self.entry_positions[tracker_id] = curr_pos
                # We need to store entry edge for relative calculation
                # Since we don't store it in a clean struct yet, let's just re-calculate on exit
                # Or better: let's store it now. 
                # But to minimize state changes, calculating both on exit (using stored entry pos) is also fine
                # provided the entry point is close enough to the edge.
                # Actually, capturing the "closest edge" at the exact moment of entry is safer 
                # than looking at entry_pos later which might be deeply inside if FPS is low.
                # Let's add a temporary dict for entry edges.
                if not hasattr(self, 'entry_edges'):
                    self.entry_edges = {}
                
                self.entry_edges[tracker_id] = self._get_closest_edge_index(curr_pos, self.zone_polygon, (w, h))
                logger.debug(f"Vehicle {tracker_id} entered zone at edge {self.entry_edges[tracker_id]}")
            
            # Zone exit - calculate direction from entry edge to exit edge
            elif not is_inside and was_inside:
                self.objects_in_zone.discard(tracker_id)
                
                if tracker_id in self.entry_positions:
                    entry_pos = self.entry_positions[tracker_id]
                    
                    # Get edges
                    entry_edge = self.entry_edges.get(tracker_id)
                    if entry_edge is None:
                        # Fallback if missed (e.g. restart)
                        entry_edge = self._get_closest_edge_index(entry_pos, self.zone_polygon, (w, h))
                        
                    exit_edge = self._get_closest_edge_index(curr_pos, self.zone_polygon, (w, h))
                    
                    turn_type, direction_deg = self._calculate_turn_from_edges(entry_edge, exit_edge, len(self.zone_polygon))
                    
                    self.completed_turns[tracker_id] = TurnResult(
                        track_id=tracker_id,
                        entry_pos=entry_pos,
                        exit_pos=curr_pos,
                        direction_deg=direction_deg, # Keeping dummy angle or specific value for logic
                        turn_type=turn_type,
                        turn_symbol=self.TURN_SYMBOLS.get(turn_type, "?"),
                        timestamp=timestamp
                    )
                    
                    logger.info(
                        f"Vehicle {tracker_id}: {turn_type.upper()} (Edge {entry_edge}->{exit_edge}) {self.TURN_SYMBOLS.get(turn_type)}"
                    )
                    
                    # Cleanup entry edge
                    if tracker_id in self.entry_edges:
                        del self.entry_edges[tracker_id]
            
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
        if hasattr(self, 'entry_edges'):
            self.entry_edges.pop(track_id, None)

    def _get_closest_edge_index(self, point: Tuple[float, float], polygon: np.ndarray, frame_size: Tuple[int, int]) -> int:
        """
        Find the index of the polygon edge closest to the point.
        Edge i connects vertex i to i+1.
        """
        w, h = frame_size
        px, py = point
        
        min_dist = float('inf')
        closest_idx = -1
        
        num_points = len(polygon)
        pixel_poly = (polygon * np.array([w, h], dtype=np.float32)).astype(np.float32)
        
        for i in range(num_points):
            p1 = pixel_poly[i]
            p2 = pixel_poly[(i + 1) % num_points]
            
            # Distance from point to line segment p1-p2
            l2 = np.sum((p1 - p2)**2)
            if l2 == 0:
                dist = np.sqrt((px - p1[0])**2 + (py - p1[1])**2)
            else:
                t = ((px - p1[0]) * (p2[0] - p1[0]) + (py - p1[1]) * (p2[1] - p1[1])) / l2
                t = max(0, min(1, t))
                proj_x = p1[0] + t * (p2[0] - p1[0])
                proj_y = p1[1] + t * (p2[1] - p1[1])
                dist = np.sqrt((px - proj_x)**2 + (py - proj_y)**2)
            
            if dist < min_dist:
                min_dist = dist
                closest_idx = i
                
        return closest_idx

    def _calculate_turn_from_edges(
        self, 
        entry_edge: int, 
        exit_edge: int, 
        num_edges: int
    ) -> Tuple[str, float]:
        """
        Calculate turn type based on entry and exit edges.
        Robust to polygon winding (CW or CCW).
        
        Logic for 4-sided polygon:
        Diff = (exit - entry) % N
        0 -> U-Turn (Same side)
        2 -> Straight (Opposite side)
        1 or 3 -> Turn (Adjacent side)
        """
        if num_edges < 3:
            return "unknown", 0.0
            
        diff = (exit_edge - entry_edge) % num_edges
        
        if diff == 0:
            return "uturn", 180.0
        
        # Determine winding order (CW or CCW)
        is_cw = self._is_clockwise(self.zone_polygon)
        
        if num_edges == 4:
            if diff == 2:
                # Opposite side is always straight regardless of winding
                return "forward", 0.0
            
            # Logic branch based on winding
            # CW: 0(Top)->1(Right). Next Edge. Driver South turns East (Left).
            # CCW: 0(Top)->1(Left). Next Edge. Driver South turns West (Right).
            
            if diff == 1: # Next Edge
                if is_cw:
                    return "left", 270.0 
                else:
                    return "right", 90.0
            elif diff == 3: # Previous Edge
                if is_cw:
                    return "right", 90.0
                else:
                    return "left", 270.0
        
        # Fallback/Generic N-gon logic
        mid = num_edges / 2
        if abs(diff - mid) < 0.5:
            return "forward", 0.0
        elif diff < mid:
            # "Forward" along polygon winding
            return "left", 270.0 if is_cw else "right", 90.0
        else:
            return "right", 90.0 if is_cw else "left", 270.0

    def _is_clockwise(self, polygon: np.ndarray) -> bool:
        """
        Determine if polygon is clockwise (CW) or counter-clockwise (CCW).
        Uses signed area formula.
        In screen coords (Y down), positive area is CW?
        Let's check: (0,0), (1,0), (1,1), (0,1).
        Edges: (1-0)*(0+0) + (1-1)*(1+0) + (0-1)*(1+1) + (0-0)*(0+1)
        = 0 + 0 + (-1*2) + 0 = -2.
        Wait. Standard Shoelace: (x2-x1)(y2+y1).
        Let's use cross product sum: (x2-x1)(y2+y1)
        Top-Left Origin:
        0,0 -> 10,0 -> 10,10 -> 0,10 (CW square)
        (10-0)(0+0) = 0
        (10-10)(10+0) = 0
        (0-10)(10+10) = -200
        (0-0)(0+10) = 0
        Sum = -200.
        So CW is Negative in this formula?
        
        Let's try standard cross product (x1*y2 - x2*y1).
        0*0 - 10*0 = 0
        10*10 - 10*0 = 100
        10*10 - 0*10 = 100
        0*0 - 0*10 = 0
        Sum = 200. Positive.
        
        Let's use `np.cross(p1, p2)` sum.
        """
        # Close the loop
        if len(polygon) < 3: return True
        
        # Standard Shoelace
        area = 0.0
        for i in range(len(polygon)):
            j = (i + 1) % len(polygon)
            area += polygon[i][0] * polygon[j][1]
            area -= polygon[j][0] * polygon[i][1]
            
        # In screen coords (Y down):
        # 0,0 -> 10,0 -> 10,10 -> 0,10 (CW)
        # 0*0 - 10*0 = 0
        # 10*10 - 10*0 = 100
        # 10*10 - 0*10 = 100
        # 0*0 - 0*10 = 0
        # Area = 200 > 0.
        
        # So CW is Positive.
        return area > 0

    
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

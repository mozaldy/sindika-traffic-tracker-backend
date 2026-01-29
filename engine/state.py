"""Centralized vehicle state management for the modular pipeline."""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any

import numpy as np
import supervision as sv

from constants import COCO_CLASSES

logger = logging.getLogger(__name__)


@dataclass
class VehicleState:
    """
    Holds all state for a single tracked vehicle.
    
    This is the central data structure that all modules read from and write to,
    enabling modular, decoupled analysis features.
    """
    # Core identification
    track_id: int
    class_id: int
    class_name: str
    first_seen: float
    last_seen: float
    
    # Bounding box (current frame)
    current_bbox: Optional[List[float]] = None
    
    # Trajectory (list of center points)
    trajectory: List[Tuple[float, float]] = field(default_factory=list)
    
    # Speed module data
    speed_kmh: Optional[float] = None
    crossing_start: Optional[float] = None
    crossing_end: Optional[float] = None
    lane_name: Optional[str] = None
    in_zone: bool = False  # Currently between entry/exit lines
    
    # Direction module data
    direction_deg: Optional[float] = None
    direction_symbol: Optional[str] = None
    
    # Plate module data
    plate_captured: bool = False
    plate_image_path: Optional[str] = None
    plate_text: Optional[str] = None
    
    # Logging flag
    logged: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "track_id": self.track_id,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "speed_kmh": self.speed_kmh,
            "direction_deg": self.direction_deg,
            "direction_symbol": self.direction_symbol,
            "lane_name": self.lane_name,
            "crossing_start": self.crossing_start,
            "crossing_end": self.crossing_end,
            "plate_text": self.plate_text,
            "plate_image_path": self.plate_image_path,
        }


class VehicleStateManager:
    """
    Central state manager for all tracked vehicles.
    
    This class maintains the lifecycle of vehicle states:
    1. Create state when a new track_id appears
    2. Update state each frame (position, trajectory)
    3. Mark as completed when analysis is done (e.g., crossed exit line)
    4. Cleanup when vehicle leaves frame
    
    Modules interact with vehicles through this manager rather than
    maintaining their own separate state dictionaries.
    """
    
    def __init__(self):
        """Initialize the state manager."""
        # Active vehicles currently in frame
        self.vehicles: Dict[int, VehicleState] = {}
        
        # Vehicles that completed crossing (pending logging)
        self.completed_vehicles: Dict[int, VehicleState] = {}
        
        # Trail history for visualization (kept separate for performance)
        self.trails: Dict[int, List[Tuple[float, float]]] = {}
        
        logger.debug("VehicleStateManager initialized")
    
    def update(
        self, 
        detections: sv.Detections, 
        timestamp: float
    ) -> None:
        """
        Update vehicle states from new detections.
        
        Args:
            detections: Current frame detections with tracker IDs
            timestamp: Current frame timestamp in seconds
        """
        if detections.tracker_id is None:
            return
        
        current_ids: Set[int] = set()
        
        for i, tracker_id in enumerate(detections.tracker_id):
            tracker_id = int(tracker_id)
            current_ids.add(tracker_id)
            
            bbox = detections.xyxy[i].tolist()
            class_id = int(detections.class_id[i]) if detections.class_id is not None else 0
            class_name = COCO_CLASSES.get(class_id, f"class_{class_id}")
            
            # Calculate center point
            center = (
                (bbox[0] + bbox[2]) / 2,
                (bbox[1] + bbox[3]) / 2
            )
            
            if tracker_id in self.vehicles:
                # Update existing vehicle
                vehicle = self.vehicles[tracker_id]
                vehicle.last_seen = timestamp
                vehicle.current_bbox = bbox
                vehicle.trajectory.append(center)
            else:
                # Create new vehicle state
                vehicle = VehicleState(
                    track_id=tracker_id,
                    class_id=class_id,
                    class_name=class_name,
                    first_seen=timestamp,
                    last_seen=timestamp,
                    current_bbox=bbox,
                    trajectory=[center]
                )
                self.vehicles[tracker_id] = vehicle
                logger.debug(f"New vehicle tracked: {tracker_id} ({class_name})")
            
            # Update trail for visualization
            if tracker_id not in self.trails:
                self.trails[tracker_id] = []
            self.trails[tracker_id].append(center)
        
        # Cleanup stale vehicles
        self._cleanup_stale(current_ids)
    
    def get_vehicle(self, track_id: int) -> Optional[VehicleState]:
        """
        Get state for a specific vehicle.
        
        Args:
            track_id: The tracker ID to look up
            
        Returns:
            VehicleState if found, None otherwise
        """
        return self.vehicles.get(track_id)
    
    def mark_completed(self, track_id: int) -> None:
        """
        Mark a vehicle as having completed analysis (e.g., crossed exit line).
        
        The vehicle remains in active tracking but is flagged for logging.
        
        Args:
            track_id: The tracker ID to mark
        """
        if track_id in self.vehicles:
            vehicle = self.vehicles[track_id]
            if track_id not in self.completed_vehicles:
                self.completed_vehicles[track_id] = vehicle
                logger.debug(f"Vehicle {track_id} marked as completed")
    
    def get_completed_unlogged(self) -> List[VehicleState]:
        """
        Get all completed vehicles that haven't been logged yet.
        
        Returns:
            List of VehicleState objects pending logging
        """
        return [
            v for v in self.completed_vehicles.values() 
            if not v.logged
        ]
    
    def mark_logged(self, track_id: int) -> None:
        """
        Mark a vehicle as logged (event saved to database).
        
        Args:
            track_id: The tracker ID to mark
        """
        if track_id in self.completed_vehicles:
            self.completed_vehicles[track_id].logged = True
    
    def _cleanup_stale(self, current_ids: Set[int]) -> None:
        """
        Remove vehicles that are no longer in frame.
        
        Args:
            current_ids: Set of tracker IDs present in current frame
        """
        stale_ids = set(self.vehicles.keys()) - current_ids
        
        for track_id in stale_ids:
            # Keep completed vehicles for a bit longer (they may need logging)
            if track_id not in self.completed_vehicles:
                self.vehicles.pop(track_id, None)
                self.trails.pop(track_id, None)
            else:
                # Remove from active but keep in completed
                self.vehicles.pop(track_id, None)
        
        # Cleanup old completed vehicles that are logged
        logged_ids = [
            tid for tid, v in self.completed_vehicles.items()
            if v.logged and tid not in current_ids
        ]
        for tid in logged_ids:
            self.completed_vehicles.pop(tid, None)
            self.trails.pop(tid, None)
    
    def get_active_trails(self) -> Dict[int, List[Tuple[float, float]]]:
        """
        Get all active trails for visualization.
        
        Returns:
            Dictionary mapping track_id to list of (x, y) points
        """
        return self.trails
    
    def get_vehicles_in_zone(self) -> Set[int]:
        """
        Get IDs of vehicles currently in a detection zone.
        
        Returns:
            Set of track_ids that are in_zone
        """
        return {tid for tid, v in self.vehicles.items() if v.in_zone}
    
    def reset(self) -> None:
        """Reset all state."""
        self.vehicles.clear()
        self.completed_vehicles.clear()
        self.trails.clear()
        logger.info("VehicleStateManager reset")

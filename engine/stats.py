"""Statistics management for traffic analysis."""

import logging
from typing import Dict, Set

import supervision as sv

from constants import COCO_CLASSES

logger = logging.getLogger(__name__)


class StatsManager:
    """
    Tracks traffic statistics including object counts.
    
    Maintains both current frame counts and cumulative unique counts
    for each detected object class.
    """
    
    def __init__(self):
        """Initialize the statistics manager."""
        self._unique_object_ids: Dict[str, Set[int]] = {}
        self._current_counts: Dict[str, int] = {}
        logger.debug("StatsManager initialized.")

    def update(self, detections: sv.Detections) -> None:
        """
        Update statistics with new detections.
        
        Args:
            detections: Current frame detections with tracker IDs
        """
        self._current_counts = {}
        
        if detections.tracker_id is None:
            return
        
        for class_id, tracker_id in zip(detections.class_id, detections.tracker_id):
            class_name = COCO_CLASSES.get(class_id, f"class_{class_id}")
            
            # Track unique objects
            if class_name not in self._unique_object_ids:
                self._unique_object_ids[class_name] = set()
            self._unique_object_ids[class_name].add(tracker_id)
            
            # Count current frame objects
            self._current_counts[class_name] = self._current_counts.get(class_name, 0) + 1

    def get_total_counts(self) -> Dict[str, int]:
        """
        Get cumulative unique object counts by class.
        
        Returns:
            Dictionary mapping class names to unique object counts
        """
        return {k: len(v) for k, v in self._unique_object_ids.items()}

    def get_current_counts(self) -> Dict[str, int]:
        """
        Get current frame object counts by class.
        
        Returns:
            Dictionary mapping class names to current frame counts
        """
        return self._current_counts.copy()
    
    def reset(self) -> None:
        """Reset all statistics."""
        self._unique_object_ids.clear()
        self._current_counts.clear()

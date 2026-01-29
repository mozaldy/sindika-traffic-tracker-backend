"""Object detection and tracking components."""

import logging
import numpy as np
import supervision as sv
from typing import List, Optional, Set, Dict

from rfdetr import RFDETRMedium

logger = logging.getLogger(__name__)


class ObjectDetector:
    """
    Object detector using RF-DETR model.
    
    Handles frame-by-frame object detection with configurable
    confidence threshold and target class filtering.
    """
    
    def __init__(
        self, 
        confidence_threshold: float = 0.5, 
        target_classes: Optional[List[int]] = None
    ):
        """
        Initialize the object detector.
        
        Args:
            confidence_threshold: Minimum confidence score for detections
            target_classes: List of class IDs to detect (None = all classes)
        """
        self.confidence_threshold = confidence_threshold
        self.target_classes = target_classes
        
        logger.info("Loading RF-DETR Medium model...")
        self.model = RFDETRMedium()
        self.model.optimize_for_inference()
        logger.info("Model loaded and optimized.")

    def detect(self, frame: np.ndarray) -> sv.Detections:
        """
        Detect objects in a frame.
        
        Args:
            frame: BGR image as numpy array
            
        Returns:
            Detections object containing bounding boxes, class IDs, and confidence scores
        """
        # Convert BGR to RGB for model
        frame_rgb = frame[:, :, ::-1].copy()
        
        detections = self.model.predict(frame_rgb, threshold=self.confidence_threshold)
        
        # Filter by target classes if specified
        if self.target_classes is not None:
            detections = detections[np.isin(detections.class_id, self.target_classes)]
        
        return detections


class ObjectTracker:
    """
    Object tracker using ByteTrack algorithm.
    
    Maintains object identity across frames by assigning
    consistent tracker IDs to detected objects.
    """
    
    def __init__(self):
        """Initialize the ByteTrack tracker."""
        self.tracker = sv.ByteTrack()
        logger.info("ByteTrack tracker initialized.")

    def update(self, detections: sv.Detections) -> sv.Detections:
        """
        Update tracker with new detections.
        
        Args:
            detections: Current frame detections
            
        Returns:
            Detections with assigned tracker IDs
        """
        return self.tracker.update_with_detections(detections)
    
    def reset(self) -> None:
        """Reset the tracker state."""
        self.tracker = sv.ByteTrack()

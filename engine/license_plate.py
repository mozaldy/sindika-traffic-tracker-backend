"""License plate detection and OCR."""

import logging
import cv2
import numpy as np
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


class LicensePlateReader:
    """
    License plate detection and OCR reader.
    
    Uses open-image-models for plate detection and EasyOCR for text reading.
    Models are lazy-loaded on first use to avoid impacting startup time.
    """
    
    def __init__(self, detection_model: str = "yolo-v9-t-384-license-plate-end2end"):
        """
        Initialize the license plate reader.
        
        Args:
            detection_model: Name of the detection model to use
        """
        self.detection_model_name = detection_model
        self._detector = None
        # OCR disabled per user request
        logger.info(f"LicensePlateReader initialized (lazy load). Model: {detection_model}")

    def _ensure_loaded(self) -> None:
        """Lazy load models on first use."""
        if self._detector is None:
            try:
                from open_image_models import LicensePlateDetector
                logger.info("Loading license plate detection model...")
                self._detector = LicensePlateDetector(detection_model=self.detection_model_name)
                logger.info("License plate detector loaded successfully.")
            except Exception as e:
                logger.error(f"Failed to load LicensePlateDetector: {e}")
                self._detector = False  # Mark as failed

    def detect_and_read(
        self, 
        vehicle_crop: np.ndarray
    ) -> Tuple[Optional[str], Optional[np.ndarray]]:
        """
        Detect license plate in vehicle crop.
        
        OCR is disabled, so the text return value will always be None.
        
        Args:
            vehicle_crop: Cropped image of the vehicle (BGR format)
            
        Returns:
            Tuple of (None, plate_crop) or (None, None) if no plate found
        """
        self._ensure_loaded()
        
        if self._detector is False or vehicle_crop is None or vehicle_crop.size == 0:
            return None, None
        
        try:
            # Detect license plates
            detections = self._detector.predict(vehicle_crop)
            
            if not detections or len(detections) == 0:
                return None, None
            
            # Take the best confidence detection
            best_detection = detections[0]
            bbox = best_detection.bounding_box
            x1, y1 = int(bbox.x1), int(bbox.y1)
            x2, y2 = int(bbox.x2), int(bbox.y2)
            
            # Clamp to image bounds
            h, w = vehicle_crop.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            if x2 <= x1 or y2 <= y1:
                return None, None
            
            plate_crop = vehicle_crop[y1:y2, x1:x2].copy()
            
            # OCR Disabled
            return None, plate_crop
            
        except Exception as e:
            logger.error(f"License plate detection failed: {e}")
            return None, None


import cv2
import supervision as sv
import numpy as np
from typing import List, Dict, Set, Deque
from rfdetr import RFDETRMedium
from rfdetr.util.coco_classes import COCO_CLASSES
from collections import deque

class ObjectDetector:
    """
    Wrapper for RF-DETR model to handle object detection.
    """
    def __init__(self, confidence_threshold: float = 0.5, target_classes: List[int] = None):
        self.confidence_threshold = confidence_threshold
        self.target_classes = target_classes
        print("Loading RF-DETR Medium model...")
        self.model = RFDETRMedium()
        self.model.optimize_for_inference()
        print("Model loaded and optimized.")

    def detect(self, frame: np.ndarray) -> sv.Detections:
        """
        Run inference on a frame and return supervision Detections.
        """
        # RF-DETR expects RGB
        frame_rgb = frame[:, :, ::-1].copy()
        detections = self.model.predict(frame_rgb, threshold=self.confidence_threshold)
        
        if self.target_classes is not None:
            # target_classes logic needs to handle that detections.class_id is a numpy array
            detections = detections[np.isin(detections.class_id, self.target_classes)]
            
        return detections

class ObjectTracker:
    """
    Wrapper for Supervision's ByteTrack/BotSort to handle multi-object tracking.
    """
    def __init__(self):
        self.tracker = sv.ByteTrack()

    def update(self, detections: sv.Detections) -> sv.Detections:
        """
        Update tracker with new detections and return detections with tracker_id.
        """
        return self.tracker.update_with_detections(detections)

class SpeedEstimator:
    """
    Estimates speed based on pixel displacement over time.
    Note: Without camera calibration, this is purely "pixels per frame" or relative speed.
    """
    def __init__(self, buffer_size: int = 5):
        # Tracker ID -> Deque of (x, y) centroids
        self.positions: Dict[int, Deque] = {}
        self.speeds: Dict[int, float] = {}
        self.buffer_size = buffer_size

    def update(self, detections: sv.Detections):
        """
        Update speed estimates for tracked objects.
        """
        if detections.tracker_id is None:
            return

        current_ids = set()
        
        for tracker_id, box in zip(detections.tracker_id, detections.xyxy):
            current_ids.add(tracker_id)
            
            # Calculate centroid
            x_center = (box[0] + box[2]) / 2
            y_center = (box[1] + box[3]) / 2
            
            if tracker_id not in self.positions:
                self.positions[tracker_id] = deque(maxlen=self.buffer_size)
                self.speeds[tracker_id] = 0.0
            
            self.positions[tracker_id].append((x_center, y_center))
            
            # Calculate speed if we have enough history
            if len(self.positions[tracker_id]) >= 2:
                # Euclidean distance between last two points
                prev_x, prev_y = self.positions[tracker_id][-2]
                dist = np.sqrt((x_center - prev_x)**2 + (y_center - prev_y)**2)
                # Simple smoothing
                self.speeds[tracker_id] = dist  # Units: Pixels/Frame

        # Cleanup old IDs
        for tid in list(self.positions.keys()):
            if tid not in current_ids:
                del self.positions[tid]
                del self.speeds[tid]

class StatsManager:
    """
    Manages counting statistics for the pipeline.
    """
    def __init__(self):
        # Class Name -> Set of Tracker IDs
        self.unique_object_ids: Dict[str, Set[int]] = {}
        # Class Name -> Current Count
        self.current_counts: Dict[str, int] = {}

    def update(self, detections: sv.Detections):
        """
        Update stats based on current frame detections.
        """
        # Reset current counts for this frame
        self.current_counts = {}
        
        if detections.tracker_id is None:
            return

        for class_id, tracker_id in zip(detections.class_id, detections.tracker_id):
            class_name = COCO_CLASSES[class_id]
            
            # Initialize if not present
            if class_name not in self.unique_object_ids:
                self.unique_object_ids[class_name] = set()
            
            # Update Unique
            self.unique_object_ids[class_name].add(tracker_id)
            
            # Update Current
            self.current_counts[class_name] = self.current_counts.get(class_name, 0) + 1

    def get_total_counts(self) -> Dict[str, int]:
        return {k: len(v) for k, v in self.unique_object_ids.items()}

    def get_current_counts(self) -> Dict[str, int]:
        return self.current_counts

class TrafficVisualizer:
    """
    Handles drawing annotations on frames.
    """
    def __init__(self):
        self.box_annotator = sv.BoxAnnotator()
        self.label_annotator = sv.LabelAnnotator()
        self.trace_annotator = sv.TraceAnnotator()

    def annotate(self, frame: np.ndarray, detections: sv.Detections, stats: StatsManager, speed_estimator: SpeedEstimator = None) -> np.ndarray:
        """
        Draw boxes, labels, traces, stats, and speed.
        """
        annotated_frame = frame.copy()

        # Draw Traces
        annotated_frame = self.trace_annotator.annotate(
            scene=annotated_frame,
            detections=detections
        )

        # Draw Boxes
        annotated_frame = self.box_annotator.annotate(
            scene=annotated_frame,
            detections=detections
        )

        # Create Labels
        labels = []
        if detections.tracker_id is not None:
            for class_id, tracker_id, score in zip(detections.class_id, detections.tracker_id, detections.confidence):
                class_name = COCO_CLASSES[class_id]
                label = f"#{tracker_id} {class_name}"
                
                # Add speed info if available
                if speed_estimator and tracker_id in speed_estimator.speeds:
                    speed = speed_estimator.speeds[tracker_id]
                    label += f" {speed:.1f} px/f"
                    
                labels.append(label)
        else:
             for class_id, score in zip(detections.class_id, detections.confidence):
                class_name = COCO_CLASSES[class_id]
                labels.append(f"{class_name} {score:.2f}")

        # Draw Labels
        annotated_frame = self.label_annotator.annotate(
            scene=annotated_frame,
            detections=detections,
            labels=labels
        )

        return annotated_frame

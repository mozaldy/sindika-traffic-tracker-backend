
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

class LineSpeedEstimator:
    """
    Estimates speed by detecting when objects cross two defined lines.
    Speed = Distance / TimeDelta.
    """
    def __init__(self):
        # Line A and Line B: each is [(x1,y1), (x2, y2)] (Normalized 0-1 if passed that way, but we usually convert later. 
        # For simplicity, we assume we receive absolute pixel coords or normalized. Let's assume normalized for config.)
        self.line1 = None
        self.line2 = None
        self.real_distance = 5.0 # meters
        
        # Tracker State
        # {tracker_id: { 'start_time': float, 'last_pos': (x,y) }}
        self.entry_times: Dict[int, float] = {}
        self.completed_speeds: Dict[int, Dict] = {} # {id: {speed: X, direction: Y}}
        
        # Store last known positions for crossing detection
        self.last_positions: Dict[int, tuple] = {}

    def set_config(self, line1: List[float], line2: List[float], distance: float):
        """
        lines as [x1, y1, x2, y2]
        """
        self.line1 = line1
        self.line2 = line2
        self.real_distance = distance
        print(f"Speed Config Updated: Dist={distance}m")

    def _ccw(self, A, B, C):
        return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

    def _intersect(self, A, B, C, D):
        """
        Return true if line segments AB and CD intersect
        """
        return self._ccw(A,C,D) != self._ccw(B,C,D) and self._ccw(A,B,C) != self._ccw(A,B,D)

    def update(self, detections: sv.Detections, frame_shape):
        """
        Update speed estimates.
        """
        if self.line1 is None or self.line2 is None:
            return

        import time
        height, width = frame_shape[:2]
        current_time = time.time()
        
        # Convert normalized lines to pixels
        l1 = [
            (self.line1[0]*width, self.line1[1]*height),
            (self.line1[2]*width, self.line1[3]*height)
        ]
        l2 = [
            (self.line2[0]*width, self.line2[1]*height),
            (self.line2[2]*width, self.line2[3]*height)
        ]

        if detections.tracker_id is None:
            return

        current_ids = set()
        
        for tracker_id, box in zip(detections.tracker_id, detections.xyxy):
            current_ids.add(tracker_id)
            
            # Centroid
            x_center = (box[0] + box[2]) / 2
            y_center = (box[1] + box[3]) / 2
            curr_pos = (x_center, y_center)
            
            if tracker_id in self.last_positions:
                prev_pos = self.last_positions[tracker_id]
                
                # Check crossing Line 1 (Start)
                if tracker_id not in self.entry_times:
                    if self._intersect(prev_pos, curr_pos, l1[0], l1[1]):
                        self.entry_times[tracker_id] = current_time
                        # print(f"Object {tracker_id} crossed Line 1 at {current_time}")

                # Check crossing Line 2 (End)
                if tracker_id in self.entry_times:
                     if self._intersect(prev_pos, curr_pos, l2[0], l2[1]):
                        start_time = self.entry_times[tracker_id]
                        time_diff = current_time - start_time
                        
                        if time_diff > 0.1: # Eliminate instant/noise crossings
                            speed_mps = self.real_distance / time_diff
                            speed_kmh = speed_mps * 3.6
                            
                            # Calculate Direction (Angle) from Start Line to End Line
                            # Vector from entry line center to current
                            dx = curr_pos[0] - prev_pos[0]
                            dy = curr_pos[1] - prev_pos[1]
                            angle = np.degrees(np.arctan2(dy, dx))
                            if angle < 0: angle += 360
                            
                            self.completed_speeds[tracker_id] = {
                                "speed": speed_kmh,
                                "direction": angle,
                                "timestamp": current_time
                            }
                            # print(f"Object {tracker_id} Finished! Speed: {speed_kmh:.2f} km/h")
                            
                            # Log to DB/Callback could happen here
                            
                            # Remove from entry times to prevent double counting
                            del self.entry_times[tracker_id]

            self.last_positions[tracker_id] = curr_pos

        # Cleanup
        # Remove old IDs
        # Keep completed speeds for display? Or clear after some time?
        # For now, keep them in 'completed_speeds' but maybe limit size
        
        missing_ids = set(self.last_positions.keys()) - current_ids
        for mid in missing_ids:
            del self.last_positions[mid]
            if mid in self.entry_times:
                del self.entry_times[mid] # Object left before finishing

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

    def annotate(self, frame: np.ndarray, detections: sv.Detections, stats: StatsManager, speed_estimator: 'LineSpeedEstimator' = None) -> np.ndarray:
        """
        Draw boxes, labels, traces, stats, and speed.
        """
        annotated_frame = frame.copy()

        # Draw Configured Lines if they exist
        if speed_estimator and speed_estimator.line1 and speed_estimator.line2:
             height, width = frame.shape[:2]
             
             # Draw Line 1 (Green)
             l1_start = (int(speed_estimator.line1[0]*width), int(speed_estimator.line1[1]*height))
             l1_end = (int(speed_estimator.line1[2]*width), int(speed_estimator.line1[3]*height))
             cv2.line(annotated_frame, l1_start, l1_end, (0, 255, 0), 2)
             cv2.putText(annotated_frame, "START", l1_start, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

             # Draw Line 2 (Red)
             l2_start = (int(speed_estimator.line2[0]*width), int(speed_estimator.line2[1]*height))
             l2_end = (int(speed_estimator.line2[2]*width), int(speed_estimator.line2[3]*height))
             cv2.line(annotated_frame, l2_start, l2_end, (0, 0, 255), 2)
             cv2.putText(annotated_frame, "END", l2_start, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
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
                # Check completed speeds first
                if speed_estimator and tracker_id in speed_estimator.completed_speeds:
                    speed_data = speed_estimator.completed_speeds[tracker_id]
                    speed = speed_data['speed']
                    label += f" {speed:.1f} km/h"
                elif speed_estimator and tracker_id in speed_estimator.entry_times:
                     label += " Measuring..."
                    
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

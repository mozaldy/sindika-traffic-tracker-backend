
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

class PolygonZoneEstimator:
    """
    Estimates speed and direction by detecting when objects ENTER and EXIT a defined polygon zone.
    Speed = Distance (Diameter/Path) / TimeDelta.
    Direction = Vector from Entry Point to Exit Point.
    """
    def __init__(self):
        # Zone points: [(x1,y1), (x2, y2), ...] (Normalized 0-1)
        self.zone_polygon = None
        self.real_distance = 5.0 # meters (approximate crossing distance)
        
        # Tracker State
        # {tracker_id: deque([(x,y), ...], maxlen=30)}
        self.position_history: Dict[int, Deque] = {}
        
        # Detection State
        self.entry_times: Dict[int, float] = {}
        self.entry_positions: Dict[int, tuple] = {}
        self.exit_times: Dict[int, float] = {}
        self.exit_positions: Dict[int, tuple] = {}
        
        self.completed_speeds: Dict[int, Dict] = {} # {id: {speed: X, direction: Y, ...}}
        
        # Track objects currently INSIDE the zone
        self.objects_in_zone: Set[int] = set()
        
        # Store last known positions
        self.last_positions: Dict[int, tuple] = {}

    def set_config(self, zone_points: List[float], distance: float):
        """
        zone_points as [x1, y1, x2, y2, ...]
        """
        # Convert flat list to list of tuples
        if zone_points and len(zone_points) >= 6: # At least 3 points
            pts = []
            for i in range(0, len(zone_points), 2):
                pts.append((zone_points[i], zone_points[i+1]))
            self.zone_polygon = np.array(pts, dtype=np.float32)
            self.real_distance = distance
            print(f"Zone Config Updated: {len(pts)} points, Dist={distance}m")
        else:
            print("Invalid zone config received")

    def _get_direction_symbol(self, angle: float) -> str:
        """
        Returns a unicode arrow symbol based on the angle (0-360).
        0 deg = East (Right)
        90 deg = South (Down)
        180 deg = West (Left)
        270 deg = North (Up)
        """
        if angle >= 337.5 or angle < 22.5:
            return "⇨"
        elif 22.5 <= angle < 67.5:
            return "⬎"
        elif 67.5 <= angle < 112.5:
            return "⇩"
        elif 112.5 <= angle < 157.5:
            return "⬐"
        elif 157.5 <= angle < 202.5:
            return "⇦"
        elif 202.5 <= angle < 247.5:
            return "⬏"
        elif 247.5 <= angle < 292.5:
            return "⇧"
        elif 292.5 <= angle < 337.5:
            return "⬑"
        return "?"

    def update(self, detections: sv.Detections, frame_shape):
        """
        Update zone status and estimations.
        """
        if self.zone_polygon is None:
            return

        import time
        height, width = frame_shape[:2]
        current_time = time.time()
        
        # Scale polygon to pixel coordinates for checking
        # self.zone_polygon is user config (0-1)
        # We need integer array for pointPolygonTest? It works with float but for consistency let's scale
        # actually pointPolygonTest works nicely with float32 if we just pass scaled coordinates
        
        zone_pixel = (self.zone_polygon * np.array([width, height], dtype=np.float32)).astype(np.int32)

        if detections.tracker_id is None:
            return

        current_ids = set()
        
        for tracker_id, box in zip(detections.tracker_id, detections.xyxy):
            current_ids.add(tracker_id)
            
            # Centroid
            x_center = (box[0] + box[2]) / 2
            y_center = (box[1] + box[3]) / 2
            curr_pos = (x_center, y_center)
            
            # Update History (Smoothing)
            if tracker_id not in self.position_history:
                self.position_history[tracker_id] = deque(maxlen=30)
            self.position_history[tracker_id].append(curr_pos)
            
            # Check Zone Status
            # measureDist=False returns +1 (inside), -1 (outside), 0 (on edge)
            result = cv2.pointPolygonTest(zone_pixel, curr_pos, False)
            is_inside = result >= 0
            
            was_inside = tracker_id in self.objects_in_zone
            
            # Event: ENTER Zone
            if is_inside and not was_inside:
                self.objects_in_zone.add(tracker_id)
                self.entry_times[tracker_id] = current_time
                self.entry_positions[tracker_id] = curr_pos
                # print(f"Object {tracker_id} ENTERED zone")

            # Event: EXIT Zone
            elif not is_inside and was_inside:
                self.objects_in_zone.remove(tracker_id)
                
                # Check if we have an entry record for this object
                if tracker_id in self.entry_times:
                    start_time = self.entry_times[tracker_id]
                    duration = current_time - start_time
                    
                    if duration > 0.5: # Minimum duration to avoid flickers
                        speed_mps = self.real_distance / duration
                        speed_kmh = speed_mps * 3.6
                        
                        start_pos = self.entry_positions[tracker_id]
                        end_pos = curr_pos # Exit point
                        
                        dx = end_pos[0] - start_pos[0]
                        dy = end_pos[1] - start_pos[1]
                        
                        angle = np.degrees(np.arctan2(dy, dx))
                        if angle < 0: angle += 360
                        
                        self.completed_speeds[tracker_id] = {
                            "speed": speed_kmh,
                            "direction": angle,
                            "direction_symbol": self._get_direction_symbol(angle),
                            "timestamp": current_time
                        }
                        # print(f"Object {tracker_id} EXITED. Speed: {speed_kmh:.1f} km/h, Dir: {angle:.0f}")

            self.last_positions[tracker_id] = curr_pos

        # Cleanup
        missing_ids = set(self.last_positions.keys()) - current_ids
        for mid in missing_ids:
            if mid in self.last_positions: del self.last_positions[mid]
            if mid in self.position_history: del self.position_history[mid]
            if mid in self.entry_times: del self.entry_times[mid]
            if mid in self.entry_positions: del self.entry_positions[mid]
            if mid in self.objects_in_zone: self.objects_in_zone.remove(mid)

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

    def annotate(self, frame: np.ndarray, detections: sv.Detections, stats: StatsManager, speed_estimator: 'PolygonZoneEstimator' = None) -> np.ndarray:
        """
        Draw boxes, labels, traces, stats, and speed.
        """
        annotated_frame = frame.copy()

        # Draw Configured Zone if it exists
        if speed_estimator and speed_estimator.zone_polygon is not None:
             height, width = frame.shape[:2]
             
             # Scale back to pixels
             zone_pixel = (speed_estimator.zone_polygon * np.array([width, height], dtype=np.float32)).astype(np.int32)
             
             # Reshape for polylines (N, 1, 2)
             zone_pixel = zone_pixel.reshape((-1, 1, 2))
             
             cv2.polylines(annotated_frame, [zone_pixel], True, (0, 255, 0), 2)
             
             # Draw Label Center
             # M = cv2.moments(zone_pixel)
             # if M["m00"] != 0:
             #    cX = int(M["m10"] / M["m00"])
             #    cY = int(M["m01"] / M["m00"])
             #    cv2.putText(annotated_frame, "ZONE", (cX-20, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
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
                    symbol = speed_data.get('direction_symbol', '')
                    label += f" {speed:.1f} km/h {symbol}"
                elif speed_estimator and tracker_id in speed_estimator.objects_in_zone:
                     label += " In Zone"
                    
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

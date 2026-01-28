
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
        # Zone polygon points
        self.zone_polygon = None
        
        # Real world approximate distance of the path through the zone (or gate)
        self.real_distance = 5.0 # meters
        self.lane_direction_degrees = 90.0 # Default down
        
        # Detection State
        self.entry_times: Dict[int, float] = {}
        self.entry_positions: Dict[int, tuple] = {}
        self.completed_speeds: Dict[int, Dict] = {} # {id: {speed: X, direction: Y, ...}}
        
        # Track objects currently INSIDE the zone
        self.objects_in_zone: Set[int] = set()
        
        # Store last known positions (helper for logic if needed)
        self.last_positions: Dict[int, tuple] = {}
        
        # Smoothing History for robust IN/OUT checks
        self.position_history: Dict[int, Deque] = {}
        
        # Active trails for visualization
        self.active_trails: Dict[int, List[tuple]] = {}

    def set_config(self, zone_points: List[float], distance: float):
        """
        zone_points: Flat list [x1, y1, x2, y2, ...] or list of tuples
        distance: Real world distance in meters
        """
        if not zone_points or len(zone_points) < 6:
            print("Invalid zone config received")
            return

        # Convert to numpy array of shape (N, 2)
        pts = []
        for i in range(0, len(zone_points), 2):
            pts.append((zone_points[i], zone_points[i+1]))
        self.zone_polygon = np.array(pts, dtype=np.float32)
        self.real_distance = distance
        
        # Calculate Lane Direction if it looks like a Quadrilateral (4 points)
        # We assume order: StartP0, StartP1, EndP2, EndP3 (or similar 8-point rect definitions)
        # Actually, standard rect order: P0, P1, P2, P3.
        # Let's assume P0-P1 is START edge and P3-P2 is END edge (standard box).
        # Midpoint Start
        if len(pts) == 4:
            mx_start = (pts[0][0] + pts[1][0]) / 2
            my_start = (pts[0][1] + pts[1][1]) / 2
            
            # Midpoint End
            mx_end = (pts[3][0] + pts[2][0]) / 2
            my_end = (pts[3][1] + pts[2][1]) / 2
            
            dx_lane = mx_end - mx_start
            dy_lane = my_end - my_start
            
            self.lane_direction_degrees = np.degrees(np.arctan2(dy_lane, dx_lane))
            if self.lane_direction_degrees < 0: self.lane_direction_degrees += 360
            print(f"Zone Config Updated: {len(pts)} points, Dist={distance}m, LaneDir={self.lane_direction_degrees:.1f}")
        else:
             print(f"Zone Config Updated: {len(pts)} points, Dist={distance}m. LaneDir not auto-calculated (using default/prev).")

    def set_config_lines(self, line1: List[float], line2: List[float], distance: float):
        """
        Backward compatibility for 2-line config (converts to Quad).
        Start: line1, End: line2.
        """
        # Form a quad: L1_P0, L1_P1, L2_P1, L2_P0 (or appropriate winding)
        # L1: x1,y1, x2,y2. L2: x3,y3, x4,y4
        # Quad: L1_start, L1_end, L2_end, L2_start
        polygon = [
            line1[0], line1[1], # P0
            line1[2], line1[3], # P1
            line2[2], line2[3], # P2 (L2 End) -- Wait, P2 usually follows P1.
            line2[0], line2[1]  # P3 (L2 Start)
        ]
        self.set_config(polygon, distance)

    def _get_direction_symbol(self, angle: float) -> str:
        """
        Returns a unicode arrow symbol based on the angle (0-360).
        Maps 0 (Relative Forward) to Up Arrow.
        180 (Relative Backward) to Down Arrow.
        THIS IS FOR RELATIVE ANGLE VISUALIZATION in console/logs that support unicode.
        """
        # Map Angle (-180 to 180) to 0-360 for symbol lookup
        # 0 -> Up (North)
        # 90 -> Right (East)
        # 180 -> Down (South)
        # -90 (270) -> Left (West)
        
        # Our Relative Angle: 0 = Forward.
        # Use simple mapping:
        # -22.5 to 22.5 -> Forward (Up)
        if -22.5 <= angle < 22.5: return "↑"
        if 22.5 <= angle < 67.5: return "↗"
        if 67.5 <= angle < 112.5: return "→"
        if 112.5 <= angle < 157.5: return "↘"
        if 157.5 <= angle <= 180 or -180 <= angle < -157.5: return "↓"
        if -157.5 <= angle < -112.5: return "↙"
        if -112.5 <= angle < -67.5: return "←"
        if -67.5 <= angle < -22.5: return "↖"
        return "?"

    def update(self, detections: sv.Detections, frame_shape, timestamp: float):
        """
        Update zone status and estimations.
        """
        if self.zone_polygon is None:
            return

        height, width = frame_shape[:2]
        current_time = timestamp
        
        # Scale polygon to pixel coordinates
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
            
            # Update Active Trails (for viz)
            if tracker_id not in self.active_trails:
                self.active_trails[tracker_id] = []
            self.active_trails[tracker_id].append(curr_pos)

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

            # Event: EXIT Zone
            elif not is_inside and was_inside:
                self.objects_in_zone.remove(tracker_id)
                
                # Check if we have an entry record
                if tracker_id in self.entry_times:
                    start_time = self.entry_times[tracker_id]
                    duration = current_time - start_time
                    
                    if duration > 0.5: # Minimum duration
                        speed_mps = self.real_distance / duration
                        speed_kmh = speed_mps * 3.6
                        
                        start_pos = self.entry_positions[tracker_id]
                        end_pos = curr_pos # Exit point (approximated as current pos just outside)
                        
                        dx = end_pos[0] - start_pos[0]
                        dy = end_pos[1] - start_pos[1]
                        
                        # Absolute Angle
                        angle_abs = np.degrees(np.arctan2(dy, dx))
                        if angle_abs < 0: angle_abs += 360
                        
                        # Relative Angle (0 = Forward/Along Lane)
                        angle_rel = angle_abs - self.lane_direction_degrees
                        # Normalize to [-180, 180]
                        while angle_rel > 180: angle_rel -= 360
                        while angle_rel < -180: angle_rel += 360
                        
                        self.completed_speeds[tracker_id] = {
                            "speed": speed_kmh,
                            "direction": angle_rel,
                            "direction_symbol": self._get_direction_symbol(angle_rel),
                            "timestamp": current_time,
                            "start_time": start_time,
                            "end_time": current_time
                        }
                        
                        # Cleanup entry data
                        del self.entry_times[tracker_id]
                        if tracker_id in self.entry_positions:
                            del self.entry_positions[tracker_id]
                        if tracker_id in self.active_trails:
                            pass # Keep trail for visualization (turn green)

            self.last_positions[tracker_id] = curr_pos

        # Cleanup lost tracks
        missing_ids = set(self.last_positions.keys()) - current_ids
        for mid in missing_ids:
            if mid in self.last_positions: del self.last_positions[mid]
            if mid in self.position_history: del self.position_history[mid]
            if mid in self.entry_times: del self.entry_times[mid]
            if mid in self.entry_positions: del self.entry_positions[mid]
            if mid in self.active_trails: del self.active_trails[mid]
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
        self.current_counts = {}
        if detections.tracker_id is None:
            return

        for class_id, tracker_id in zip(detections.class_id, detections.tracker_id):
            class_name = COCO_CLASSES[class_id]
            
            if class_name not in self.unique_object_ids:
                self.unique_object_ids[class_name] = set()
            self.unique_object_ids[class_name].add(tracker_id)
            
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

    def annotate(self, frame: np.ndarray, detections: sv.Detections, stats: StatsManager, speed_estimator: 'PolygonZoneEstimator' = None, timestamp_info: str = None) -> np.ndarray:
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
             
             # Identify Start/End for visual clarity if it's a quad
             if len(zone_pixel) == 4:
                 pts = zone_pixel.reshape(4, 2)
                 # Start Line (P0-P1)
                 cv2.line(annotated_frame, tuple(pts[0]), tuple(pts[1]), (0, 255, 0), 2)
                 cv2.putText(annotated_frame, "START", tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                 
                 # End Line (P3-P2)
                 cv2.line(annotated_frame, tuple(pts[3]), tuple(pts[2]), (0, 0, 255), 2)
                 cv2.putText(annotated_frame, "END", tuple(pts[3]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Draw Active Trails
        if speed_estimator:
            for tid, trail in speed_estimator.active_trails.items():
                if len(trail) > 1:
                    color = None
                    thickness = 2
                    
                    # Logic: Only draw if passing through (In Zone) or Completed
                    if tid in speed_estimator.completed_speeds:
                        color = (0, 255, 0) # Green for Success
                        thickness = 3
                    elif tid in speed_estimator.objects_in_zone:
                         color = (0, 255, 255) # Yellow for In-Progress
                         
                    if color:
                        cv2.polylines(annotated_frame, [np.array(trail, dtype=np.int32)], False, color, thickness)

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
                if speed_estimator:
                    if tracker_id in speed_estimator.completed_speeds:
                        speed_data = speed_estimator.completed_speeds[tracker_id]
                        speed = speed_data['speed']
                        symbol = speed_data.get('direction_symbol', '')
                        label += f" {speed:.1f} km/h {symbol}"
                    elif tracker_id in speed_estimator.objects_in_zone:
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

        # Draw Timestamp
        if timestamp_info:
            cv2.putText(annotated_frame, timestamp_info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(annotated_frame, timestamp_info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1, cv2.LINE_AA)

        return annotated_frame

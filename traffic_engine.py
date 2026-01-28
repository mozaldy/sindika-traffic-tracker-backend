import cv2
import math
import supervision as sv
import numpy as np
from typing import List, Dict, Set, Deque
from rfdetr import RFDETRMedium
from rfdetr.util.coco_classes import COCO_CLASSES
from collections import deque

class ObjectDetector:
    def __init__(self, confidence_threshold: float = 0.5, target_classes: List[int] = None):
        self.confidence_threshold = confidence_threshold
        self.target_classes = target_classes
        print("Loading RF-DETR Medium model...")
        self.model = RFDETRMedium()
        self.model.optimize_for_inference()
        print("Model loaded and optimized.")

    def detect(self, frame: np.ndarray) -> sv.Detections:
        frame_rgb = frame[:, :, ::-1].copy()
        detections = self.model.predict(frame_rgb, threshold=self.confidence_threshold)
        if self.target_classes is not None:
            detections = detections[np.isin(detections.class_id, self.target_classes)]
        return detections

class ObjectTracker:
    def __init__(self):
        self.tracker = sv.ByteTrack()

    def update(self, detections: sv.Detections) -> sv.Detections:
        return self.tracker.update_with_detections(detections)

class StatsManager:
    def __init__(self):
        self.unique_object_ids: Dict[str, Set[int]] = {}
        self.current_counts: Dict[str, int] = {}

    def update(self, detections: sv.Detections):
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

class PolygonZoneEstimator:
    def __init__(self):
        self.zone_polygon = None
        self.real_distance = 5.0 
        self.position_history: Dict[int, Deque] = {}
        self.entry_times: Dict[int, float] = {}
        self.entry_positions: Dict[int, tuple] = {}
        self.entry_edges: Dict[int, int] = {} 
        self.completed_speeds: Dict[int, Dict] = {} 
        self.objects_in_zone: Set[int] = set()
        self.last_positions: Dict[int, tuple] = {}
        self.active_trails: Dict[int, List[tuple]] = {}

    def set_config(self, zone_points: List[float], distance: float):
        if zone_points and len(zone_points) >= 6:
            pts = []
            for i in range(0, len(zone_points), 2):
                pts.append((zone_points[i], zone_points[i+1]))
            self.zone_polygon = np.array(pts, dtype=np.float32)
            self.real_distance = distance
        else:
            print("Invalid zone config")

    def set_config_lines(self, line1: List[float], line2: List[float], distance: float):
        # Fallback for line-based config: construct a polygon from the two lines
        if len(line1) == 4 and len(line2) == 4:
            # Connect the lines to form a polygon. Order: Line1 -> Line2 (reverse)
            # Line1: p1 -> p2. Line2: p3 -> p4.
            # Polygon: p1, p2, p4, p3 ? Or p1, p2, p3, p4?
            # Standard "gate" usually implies Line 1 is Entry, Line 2 is Exit?
            # We'll just take the 4 points.
            pts = [line1[0], line1[1], line1[2], line1[3],
                   line2[2], line2[3], line2[0], line2[1]] # Reversed line2 for loop?
            
            # Simple list concatenation for now, assuming user will switch to polygon
            pts = line1 + [line2[2], line2[3], line2[0], line2[1]]
            self.set_config(pts, distance)

    def _ccw(self, A, B, C):
        return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

    def _intersect(self, A, B, C, D):
        return self._ccw(A,C,D) != self._ccw(B,C,D) and self._ccw(A,B,C) != self._ccw(A,B,D)

    def _get_crossed_edge(self, prev_pos, curr_pos, frame_shape):
        if self.zone_polygon is None: return None
        h, w = frame_shape[:2]
        poly_pixels = (self.zone_polygon * np.array([w, h], dtype=np.float32)).astype(np.int32)
        for i in range(len(poly_pixels)):
            p1, p2 = poly_pixels[i], poly_pixels[(i + 1) % len(poly_pixels)]
            if self._intersect(prev_pos, curr_pos, p1, p2):
                return i
        return None

    def _get_line_direction_symbol(self, entry_idx: int, exit_idx: int) -> str:
        """ 
        Mapping simbol lama yang mencakup arah belok (tikungan).
        0=Top, 1=Right, 2=Bottom, 3=Left (tergantung urutan penggambaran poligon)
        """
        mapping = {
            (0, 2): "⇩", (2, 0): "⇧", (1, 3): "⇦", (3, 1): "⇨",
            (0, 1): "↳", (1, 0): "⬑", # Top ke Right, Right ke Top
            (0, 3): "↲", (3, 0): "⬏", # Top ke Left, Left ke Top
            (2, 1): "↱", (1, 2): "⬐", # Bottom ke Right, Right ke Bottom
            (2, 3): "↰", (3, 2): "⬎", # Bottom ke Left, Left ke Bottom
        }
        return mapping.get((entry_idx, exit_idx), "?")

    def update(self, detections: sv.Detections, frame_shape, timestamp: float):
        if self.zone_polygon is None or detections.tracker_id is None:
            return
        h, w = frame_shape[:2]
        zone_pixel = (self.zone_polygon * np.array([w, h], dtype=np.float32)).astype(np.int32)
        current_ids = set()
        
        for tracker_id, box in zip(detections.tracker_id, detections.xyxy):
            current_ids.add(tracker_id)
            curr_pos = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
            
            if tracker_id not in self.active_trails:
                self.active_trails[tracker_id] = []
            self.active_trails[tracker_id].append(curr_pos)
            
            is_inside = cv2.pointPolygonTest(zone_pixel, curr_pos, False) >= 0
            was_inside = tracker_id in self.objects_in_zone
            
            if is_inside and not was_inside:
                self.objects_in_zone.add(tracker_id)
                self.entry_times[tracker_id] = timestamp
                self.entry_positions[tracker_id] = curr_pos
                if tracker_id in self.last_positions:
                    edge = self._get_crossed_edge(self.last_positions[tracker_id], curr_pos, frame_shape)
                    if edge is not None: self.entry_edges[tracker_id] = edge
            
            elif not is_inside and was_inside:
                self.objects_in_zone.remove(tracker_id)
                if tracker_id in self.entry_times:
                    duration = timestamp - self.entry_times[tracker_id]
                    if duration > 0.3:
                        speed_kmh = (self.real_distance / duration) * 3.6
                        exit_edge = self._get_crossed_edge(self.last_positions[tracker_id], curr_pos, frame_shape)
                        entry_edge = self.entry_edges.get(tracker_id)
                        
                        # Menggunakan mapping simbol belok dari logika lama
                        symbol = self._get_line_direction_symbol(entry_edge, exit_edge) if entry_edge is not None and exit_edge is not None else "?"
                        
                        entry_pos = self.entry_positions.get(tracker_id)
                        angle = 0.0
                        if entry_pos:
                            dx = curr_pos[0] - entry_pos[0]
                            dy = curr_pos[1] - entry_pos[1]
                            angle = math.degrees(math.atan2(dy, dx)) % 360

                        self.completed_speeds[tracker_id] = {
                            "speed": speed_kmh,
                            "direction": angle,
                            "direction_symbol": symbol,
                            "timestamp": timestamp,
                            "start_time": self.entry_times.get(tracker_id, 0.0),
                            "end_time": timestamp
                        }
            
            self.last_positions[tracker_id] = curr_pos

        for mid in list(self.last_positions.keys() - current_ids):
            for d in [self.last_positions, self.entry_times, self.entry_positions, self.entry_edges, self.active_trails]:
                if mid in d: del d[mid]
            if mid in self.objects_in_zone: self.objects_in_zone.remove(mid)

class TrafficVisualizer:
    def __init__(self):
        self.box_annotator = sv.BoxAnnotator()
        self.label_annotator = sv.LabelAnnotator()

    def annotate(self, frame: np.ndarray, detections: sv.Detections, stats, speed_estimator: PolygonZoneEstimator, timestamp_info: str = None) -> np.ndarray:
        annotated_frame = frame.copy()
        h, w = frame.shape[:2]
        
        # Draw Zone & Edge Numbers
        if speed_estimator.zone_polygon is not None:
            zone_pixel = (speed_estimator.zone_polygon * np.array([w, h], dtype=np.float32)).astype(np.int32)
            cv2.polylines(annotated_frame, [zone_pixel.reshape((-1, 1, 2))], True, (0, 255, 0), 2)
            for i in range(len(zone_pixel)):
                p1, p2 = zone_pixel[i], zone_pixel[(i + 1) % len(zone_pixel)]
                mid = (int((p1[0]+p2[0])/2), int((p1[1]+p2[1])/2))
                cv2.circle(annotated_frame, mid, 12, (0, 0, 0), -1)
                cv2.putText(annotated_frame, str(i+1), (mid[0]-6, mid[1]+6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Draw Trails
        for tid, trail in speed_estimator.active_trails.items():
            if len(trail) > 1:
                color = (0, 255, 0) if tid in speed_estimator.completed_speeds else (0, 255, 255)
                cv2.polylines(annotated_frame, [np.array(trail, dtype=np.int32)], False, color, 2)

        # Create Labels with Turning Symbols
        labels = []
        if detections.tracker_id is not None:
            for class_id, tid in zip(detections.class_id, detections.tracker_id):
                name = COCO_CLASSES[class_id]
                label = f"#{tid} {name}"
                if tid in speed_estimator.completed_speeds:
                    res = speed_estimator.completed_speeds[tid]
                    # Menampilkan simbol belok dan nilai derajat
                    label += f" {res['speed']:.1f}km/h {res['direction_symbol']} ({int(res['direction'])}deg)"
                elif tid in speed_estimator.objects_in_zone:
                    label += " [In Zone]"
                labels.append(label)

        annotated_frame = self.box_annotator.annotate(scene=annotated_frame, detections=detections)
        annotated_frame = self.label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
        return annotated_frame
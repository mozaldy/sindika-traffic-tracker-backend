"""Traffic visualization with annotations."""

import cv2
import numpy as np
import supervision as sv
from typing import Optional, List, Dict, Any, TYPE_CHECKING

from constants import COCO_CLASSES

if TYPE_CHECKING:
    from .speed import MultiLaneSpeedEstimator
    from .turn import TurnDetector


class TrafficVisualizer:
    """
    Annotates video frames with detection results and lane visualizations.
    
    Draws bounding boxes, labels, lane lines, and vehicle trails on frames.
    """
    
    # Colors for different lanes (BGR format)
    LANE_COLORS = [
        ((255, 255, 0), (255, 0, 255)),  # Cyan / Magenta
        ((0, 255, 0), (0, 255, 255)),     # Green / Yellow
        ((107, 107, 255), (196, 205, 78)), # Red / Teal
        ((247, 85, 168), (22, 115, 249)),  # Purple / Orange
    ]
    
    # Zone colors by type (BGR format)
    ZONE_COLORS = {
        "direction": (0, 255, 0),    # Green
        "plate": (255, 100, 0),      # Blue
        "default": (0, 255, 255),    # Yellow
    }
    
    # Colors by vehicle class name (BGR format)
    CLASS_COLORS = {
        "car": (255, 144, 30),       # Dodger Blue
        "truck": (0, 165, 255),      # Orange
        "bus": (147, 20, 255),       # Deep Pink
        "motorcycle": (0, 255, 127), # Spring Green
        "bicycle": (255, 255, 0),    # Cyan
        "person": (180, 105, 255),   # Hot Pink
    }
    DEFAULT_CLASS_COLOR = (128, 128, 128)  # Gray for unknown classes
    
    TRAIL_COLOR_COMPLETED = (0, 255, 0)   # Green
    TRAIL_COLOR_IN_ZONE = (255, 0, 0)     # Blue
    TRAIL_COLOR_DEFAULT = (0, 255, 255)   # Yellow
    PLATE_LINE_COLOR = (255, 0, 255)      # Magenta/Purple for plate capture line
    
    def __init__(self):
        """Initialize the visualizer with annotation tools."""
        # Use ColorLookup.CLASS for per-class coloring
        self.box_annotator = sv.BoxAnnotator(
            color=sv.ColorPalette.from_hex([
                "#1E90FF",  # car - Dodger Blue
                "#FFA500",  # truck - Orange  
                "#FF1493",  # bus - Deep Pink
                "#00FF7F",  # motorcycle - Spring Green
                "#00FFFF",  # bicycle - Cyan
                "#FF69B4",  # person - Hot Pink
                "#9370DB",  # extra - Medium Purple
                "#20B2AA",  # extra - Light Sea Green
            ]),
            color_lookup=sv.ColorLookup.CLASS
        )
        self.label_annotator = sv.LabelAnnotator(
            color=sv.ColorPalette.from_hex([
                "#1E90FF",  # car
                "#FFA500",  # truck
                "#FF1493",  # bus
                "#00FF7F",  # motorcycle
                "#00FFFF",  # bicycle
                "#FF69B4",  # person
                "#9370DB",  # extra
                "#20B2AA",  # extra
            ]),
            color_lookup=sv.ColorLookup.CLASS
        )
        self.zones_config: List[Dict[str, Any]] = []  # Store zones for multi-zone rendering
        self.plate_line: Optional[List[float]] = None  # Plate capture line

    def annotate(
        self, 
        frame: np.ndarray, 
        detections: sv.Detections, 
        stats: any,
        speed_estimator: "MultiLaneSpeedEstimator",
        turn_detector: Optional["TurnDetector"] = None,
        timestamp_info: Optional[str] = None
    ) -> np.ndarray:
        """
        Annotate a frame with detection results and visualizations.
        
        Args:
            frame: Original BGR frame
            detections: Current frame detections
            stats: Statistics manager (unused, kept for API compatibility)
            speed_estimator: Speed estimator with lane and trail data
            timestamp_info: Optional timestamp string to display
            
        Returns:
            Annotated frame
        """
        annotated = frame.copy()
        h, w = frame.shape[:2]
        
        # Draw all configured zones with type-based colors
        self._draw_all_zones(annotated, w, h)
        
        # Draw plate capture line
        self._draw_plate_line(annotated, w, h)
        
        # Draw turn detector zone (if active and not already in zones_config)
        if turn_detector:
            self._draw_zone(annotated, turn_detector, w, h)
            
        # Draw vehicle trails
        self._draw_trails(annotated, speed_estimator, turn_detector)
        
        # Create labels and annotate detections
        labels = self._create_labels(detections, speed_estimator, turn_detector)
        annotated = self.box_annotator.annotate(scene=annotated, detections=detections)
        annotated = self.label_annotator.annotate(
            scene=annotated, 
            detections=detections, 
            labels=labels
        )
        
        return annotated

    def _draw_lanes(
        self, 
        frame: np.ndarray, 
        lanes: list, 
        width: int, 
        height: int
    ) -> None:
        """Draw lane lines and ROI on the frame."""
        overlay = frame.copy()
        
        for lane_idx, lane in enumerate(lanes):
            lane_name = lane.get('name', f'Lane {lane_idx + 1}')
            line_a = lane.get('line_a')
            line_b = lane.get('line_b')
            distance = lane.get('distance', 5.0)
            
            if not (line_a and len(line_a) == 4 and line_b and len(line_b) == 4):
                continue

            colors = self.LANE_COLORS[lane_idx % len(self.LANE_COLORS)]
            
            # Points for Line A
            pt_a1 = (int(line_a[0] * width), int(line_a[1] * height))
            pt_a2 = (int(line_a[2] * width), int(line_a[3] * height))
            
            # Points for Line B 
            pt_b1 = (int(line_b[0] * width), int(line_b[1] * height))
            pt_b2 = (int(line_b[2] * width), int(line_b[3] * height))
            
            # Draw ROI (Polygon formed by A1, A2, B2, B1)
            roi_points = np.array([pt_a1, pt_a2, pt_b2, pt_b1], dtype=np.int32)
            cv2.fillPoly(overlay, [roi_points], (255, 255, 255))
            
            # Draw Line A (entry)
            cv2.line(frame, pt_a1, pt_a2, colors[0], 2)
            self._put_label(frame, f"{lane_name} A", pt_a1, pt_a2, colors[0])
            
            # Draw Line B (exit)
            cv2.line(frame, pt_b1, pt_b2, colors[1], 2)
            self._put_label(frame, f"{lane_name} B", pt_b1, pt_b2, colors[1])
            
            # Draw Dashed Side Lines (A1->B1 and A2->B2)
            self._draw_dashed_line(frame, pt_a1, pt_b1, (255, 255, 255), 1)
            self._draw_dashed_line(frame, pt_a2, pt_b2, (255, 255, 255), 1)
            
            # Show Distance on side lines
            mid_side1 = ((pt_a1[0] + pt_b1[0]) // 2, (pt_a1[1] + pt_b1[1]) // 2)
            cv2.putText(
                frame, f"{distance}m", (mid_side1[0] - 20, mid_side1[1]), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1
            )
            
        # Apply opacity overlay
        cv2.addWeighted(overlay, 0.1, frame, 0.9, 0, frame)

    def _draw_zone(
        self,
        frame: np.ndarray,
        turn_detector: "TurnDetector",
        width: int,
        height: int
    ) -> None:
        """Draw turn detection zone."""
        polygon = turn_detector.get_zone_polygon_pixels((height, width))
        if polygon is not None:
            # Draw polygon
            cv2.polylines(frame, [polygon.reshape((-1, 1, 2))], True, (0, 255, 0), 2)
            
            # Draw edge numbers
            for i in range(len(polygon)):
                p1 = polygon[i]
                p2 = polygon[(i + 1) % len(polygon)]
                mid = (int((p1[0] + p2[0]) / 2), int((p1[1] + p2[1]) / 2))
                
                # Draw white circle background
                cv2.circle(frame, mid, 12, (255, 255, 255), -1)
                # Draw number
                cv2.putText(
                    frame, str(i + 1), (mid[0] - 5, mid[1] + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2
                )
    
    def set_zones(self, zones: List[Dict[str, Any]]) -> None:
        """Set zones configuration for multi-zone rendering."""
        self.zones_config = zones

    def set_plate_line(self, line: Optional[List[float]]) -> None:
        """Set plate capture line configuration."""
        self.plate_line = line

    def _draw_plate_line(self, frame: np.ndarray, width: int, height: int) -> None:
        """Draw the plate capture line."""
        if not self.plate_line or len(self.plate_line) != 4:
            return
        
        x1 = int(self.plate_line[0] * width)
        y1 = int(self.plate_line[1] * height)
        x2 = int(self.plate_line[2] * width)
        y2 = int(self.plate_line[3] * height)
        
        # Draw thick magenta line
        cv2.line(frame, (x1, y1), (x2, y2), self.PLATE_LINE_COLOR, 3)
        
        # Draw label at midpoint
        mid_x = (x1 + x2) // 2
        mid_y = (y1 + y2) // 2
        cv2.putText(
            frame, "PLATE CAPTURE", (mid_x - 50, mid_y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.PLATE_LINE_COLOR, 2
        )
    
    def _draw_all_zones(self, frame: np.ndarray, width: int, height: int) -> None:
        """Draw all configured zones with type-based colors."""
        for zone in self.zones_config:
            zone_type = zone.get("type", "direction")
            polygon_data = zone.get("polygon", [])
            zone_name = zone.get("name", "")
            
            if not polygon_data or len(polygon_data) < 6:  # Need at least 3 points
                continue
            
            # Convert flat [x1,y1,x2,y2,...] to points array
            points = []
            for i in range(0, len(polygon_data), 2):
                if i + 1 < len(polygon_data):
                    x = int(polygon_data[i] * width)
                    y = int(polygon_data[i + 1] * height)
                    points.append([x, y])
            
            if len(points) < 3:
                continue
                
            polygon = np.array(points, dtype=np.int32)
            color = self.ZONE_COLORS.get(zone_type, self.ZONE_COLORS["default"])
            
            # Draw filled polygon with transparency
            overlay = frame.copy()
            cv2.fillPoly(overlay, [polygon], color)
            cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
            
            # Draw polygon outline
            cv2.polylines(frame, [polygon.reshape((-1, 1, 2))], True, color, 2)
            
            # Draw zone name at centroid
            if zone_name:
                centroid_x = int(sum(p[0] for p in points) / len(points))
                centroid_y = int(sum(p[1] for p in points) / len(points))
                cv2.putText(
                    frame, zone_name, (centroid_x - 30, centroid_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
                )

    def _put_label(self, frame, text, pt1, pt2, color):
        """Helper to put label on a line."""
        mid = ((pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2)
        cv2.putText(
            frame, text, (mid[0] - 20, mid[1] - 5), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1
        )

    def _draw_dashed_line(self, img, pt1, pt2, color, thickness=1, gap=10):
        """Draw a dashed line."""
        dist = np.hypot(pt2[0] - pt1[0], pt2[1] - pt1[1])
        if dist == 0:
            return
            
        dashes = int(dist / gap)
        for i in range(dashes):
            start = (
                int(pt1[0] + (pt2[0] - pt1[0]) * i / dashes),
                int(pt1[1] + (pt2[1] - pt1[1]) * i / dashes)
            )
            end = (
                int(pt1[0] + (pt2[0] - pt1[0]) * (i + 0.5) / dashes),
                int(pt1[1] + (pt2[1] - pt1[1]) * (i + 0.5) / dashes)
            )
            cv2.line(img, start, end, color, thickness)

    def _draw_trails(
        self, 
        frame: np.ndarray, 
        speed_estimator: "MultiLaneSpeedEstimator",
        turn_detector: Optional["TurnDetector"] = None
    ) -> None:
        """Draw vehicle movement trails."""
        for tid, trail in speed_estimator.active_trails.items():
            if len(trail) > 1:
                # Determine color based on state
                if tid in speed_estimator.completed_speeds:
                    color = self.TRAIL_COLOR_COMPLETED
                elif tid in speed_estimator.objects_in_lanes:
                    color = self.TRAIL_COLOR_IN_ZONE
                else:
                    color = self.TRAIL_COLOR_DEFAULT
                
                points = np.array(trail, dtype=np.int32)
                cv2.polylines(frame, [points], False, color, 2)
                
        # Also draw trails from turn detector if speed estimator didn't cover them
        if turn_detector:
            for tid, trail in turn_detector.active_trails.items():
                if tid not in speed_estimator.active_trails and len(trail) > 1:
                    points = np.array(trail, dtype=np.int32)
                    cv2.polylines(frame, [points], False, self.TRAIL_COLOR_DEFAULT, 2)

    def _create_labels(
        self, 
        detections: sv.Detections, 
        speed_estimator: "MultiLaneSpeedEstimator",
        turn_detector: Optional["TurnDetector"] = None
    ) -> list:
        """Create simple labels for each detected object (name and ID only)."""
        labels = []
        
        if detections.tracker_id is None:
            return labels
        
        for class_id, tid in zip(detections.class_id, detections.tracker_id):
            name = COCO_CLASSES.get(class_id, f"class_{class_id}")
            label = f"#{tid} {name}"
            labels.append(label)
        
        return labels

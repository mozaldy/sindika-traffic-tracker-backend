import sqlite3
import os
import cv2
import time
import logging
import numpy as np
from datetime import datetime

logger = logging.getLogger("db_manager")

class DatabaseManager:
    def __init__(self, db_path="traffic_data.db", captures_dir="captures"):
        self.db_path = db_path
        self.captures_dir = captures_dir
        
        if not os.path.exists(self.captures_dir):
            os.makedirs(self.captures_dir)
            
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS traffic_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                class_name TEXT,
                speed_kmh REAL,
                direction_deg REAL,
                image_path TEXT,
                video_source TEXT
            )
        ''')
        conn.commit()
        conn.close()

    def log_event(self, frame: np.ndarray, bbox: list, class_name: str, speed: float, direction: float, video_source="unknown"):
        """
        Logs an event: Crops image, saves to disk, inserts into DB.
        bbox: [x1, y1, x2, y2]
        """
        try:
            timestamp = time.time()
            dt_str = datetime.fromtimestamp(timestamp).strftime("%Y%m%d_%H%M%S_%f")
            
            # Crop Image
            h, w = frame.shape[:2]
            x1, y1, x2, y2 = map(int, bbox)
            # Clamp coordinates
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            if x2 > x1 and y2 > y1:
                crop = frame[y1:y2, x1:x2]
                filename = f"{dt_str}_{class_name}_{speed:.0f}kmh.jpg"
                file_path = os.path.join(self.captures_dir, filename)
                cv2.imwrite(file_path, crop)
            else:
                file_path = "" # Invalid crop

            # Insert into DB
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            c.execute('''
                INSERT INTO traffic_events (timestamp, class_name, speed_kmh, direction_deg, image_path, video_source)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (timestamp, class_name, speed, direction, file_path, video_source))
            conn.commit()
            conn.close()
            
            logger.info(f"Logged event: {class_name} @ {speed:.1f} km/h")
            
        except Exception as e:
            logger.error(f"Failed to log event: {e}")

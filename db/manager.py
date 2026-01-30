"""Database management for traffic events."""

import sqlite3
import os
import cv2
import time
import logging
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from contextlib import contextmanager

logger = logging.getLogger(__name__)


@dataclass
class TrafficEvent:
    """Represents a traffic event record."""
    id: Optional[int] = None
    timestamp: float = 0.0
    class_name: str = ""
    speed_kmh: float = 0.0
    direction_deg: float = 0.0
    direction_symbol: Optional[str] = None
    entry_edge: Optional[int] = None
    exit_edge: Optional[int] = None
    image_path: str = ""
    video_source: str = "unknown"
    crossing_start: float = 0.0
    crossing_end: float = 0.0
    license_plate: Optional[str] = None
    plate_image_path: Optional[str] = None


class DatabaseManager:
    """
    Manages traffic event storage with SQLite.
    
    Handles event logging, retrieval, and deletion with automatic
    image storage for vehicle and license plate captures.
    """
    
    # Schema version for migrations
    SCHEMA_VERSION = 3
    
    # Required columns for the traffic_events table
    REQUIRED_COLUMNS = [
        'direction_symbol',
        'crossing_start', 
        'crossing_end',
        'license_plate',
        'plate_image_path',
        'entry_edge',
        'exit_edge'
    ]
    
    def __init__(self, db_path: str = "traffic_data.db", captures_dir: str = "captures"):
        """
        Initialize the database manager.
        
        Args:
            db_path: Path to SQLite database file
            captures_dir: Directory to store captured images
        """
        self.db_path = db_path
        self.captures_dir = captures_dir
        
        self._ensure_captures_dir()
        self._init_db()
        
        logger.info(f"DatabaseManager initialized. DB: {db_path}, Captures: {captures_dir}")

    def _ensure_captures_dir(self) -> None:
        """Ensure captures directory exists."""
        if not os.path.exists(self.captures_dir):
            os.makedirs(self.captures_dir)
            logger.info(f"Created captures directory: {self.captures_dir}")

    @contextmanager
    def _get_connection(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self) -> None:
        """Initialize database schema with migrations."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Create table if not exists
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS traffic_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL,
                    class_name TEXT,
                    speed_kmh REAL,
                    direction_deg REAL,
                    direction_symbol TEXT,
                    entry_edge INTEGER,
                    exit_edge INTEGER,
                    image_path TEXT,
                    video_source TEXT,
                    crossing_start REAL,
                    crossing_end REAL,
                    license_plate TEXT,
                    plate_image_path TEXT
                )
            ''')
            
            # Run migrations
            self._run_migrations(cursor)

    def _run_migrations(self, cursor: sqlite3.Cursor) -> None:
        """Run necessary schema migrations."""
        cursor.execute("PRAGMA table_info(traffic_events)")
        existing_columns = {info[1] for info in cursor.fetchall()}
        
        migrations = {
            'direction_symbol': 'ALTER TABLE traffic_events ADD COLUMN direction_symbol TEXT',
            'crossing_start': 'ALTER TABLE traffic_events ADD COLUMN crossing_start REAL',
            'crossing_end': 'ALTER TABLE traffic_events ADD COLUMN crossing_end REAL',
            'license_plate': 'ALTER TABLE traffic_events ADD COLUMN license_plate TEXT',
            'plate_image_path': 'ALTER TABLE traffic_events ADD COLUMN plate_image_path TEXT',
            'entry_edge': 'ALTER TABLE traffic_events ADD COLUMN entry_edge INTEGER',
            'exit_edge': 'ALTER TABLE traffic_events ADD COLUMN exit_edge INTEGER',
        }
        
        for column, sql in migrations.items():
            if column not in existing_columns:
                try:
                    cursor.execute(sql)
                    logger.info(f"Migration: Added column '{column}'")
                except sqlite3.OperationalError:
                    pass

    def save_captures(
        self,
        frame: np.ndarray,
        bbox: List[float],
        class_name: str,
        speed: float,
        plate_crop: Optional[np.ndarray] = None
    ) -> Tuple[str, str]:
        """
        Save vehicle and plate images immediately.
        
        Args:
            frame: Full video frame
            bbox: Bounding box [x1, y1, x2, y2]
            class_name: Detected object class
            speed: Calculated speed
            plate_crop: Cropped license plate image
            
        Returns:
            Tuple of (vehicle_image_path, plate_image_path)
        """
        try:
            timestamp = time.time()
            dt_str = datetime.fromtimestamp(timestamp).strftime("%Y%m%d_%H%M%S_%f")
            
            image_path = self._save_vehicle_crop(frame, bbox, dt_str, class_name, speed)
            plate_image_path = self._save_plate_crop(plate_crop, dt_str)
            
            return image_path, plate_image_path
        except Exception as e:
            logger.error(f"Failed to save captures: {e}")
            return "", ""

    def log_event(
        self,
        frame: np.ndarray,
        bbox: List[float],
        class_name: str,
        speed: float,
        direction: float,
        direction_symbol: Optional[str] = None,
        entry_edge: Optional[int] = None,
        exit_edge: Optional[int] = None,
        video_source: str = "unknown",
        crossing_start: float = 0.0,
        crossing_end: float = 0.0,
        license_plate: Optional[str] = None,
        plate_crop: Optional[np.ndarray] = None,
        image_path: Optional[str] = None,
        plate_image_path: Optional[str] = None
    ) -> Optional[int]:
        """
        Log a traffic event with vehicle image capture.
        
        Args:
            frame: Full video frame
            bbox: Bounding box [x1, y1, x2, y2]
            class_name: Detected object class
            speed: Calculated speed in km/h
            direction: Direction angle in degrees
            direction_symbol: Unicode direction symbol
            entry_edge: Edge number the vehicle entered zone (1-indexed)
            exit_edge: Edge number the vehicle exited zone (1-indexed)
            video_source: Source video identifier
            crossing_start: Timestamp when crossing started
            crossing_end: Timestamp when crossing ended
            license_plate: Detected license plate text
            plate_crop: Cropped license plate image (used if plate_image_path not provided)
            image_path: Pre-saved vehicle image path (optional)
            plate_image_path: Pre-saved plate image path (optional)
            
        Returns:
            Event ID if successful, None otherwise
        """
        try:
            timestamp = time.time()
            dt_str = datetime.fromtimestamp(timestamp).strftime("%Y%m%d_%H%M%S_%f")
            
            # Use pre-saved paths or save now
            final_image_path = image_path if image_path else self._save_vehicle_crop(frame, bbox, dt_str, class_name, speed)
            final_plate_path = plate_image_path if plate_image_path else self._save_plate_crop(plate_crop, dt_str)
            
            # Insert record
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO traffic_events 
                    (timestamp, class_name, speed_kmh, direction_deg, direction_symbol, 
                     entry_edge, exit_edge, image_path, video_source, crossing_start, 
                     crossing_end, license_plate, plate_image_path)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    timestamp, class_name, float(speed), float(direction),
                    direction_symbol, entry_edge, exit_edge, final_image_path, 
                    video_source, crossing_start, crossing_end, license_plate, final_plate_path
                ))
                event_id = cursor.lastrowid
            
            logger.info(f"Logged event #{event_id}: {class_name} @ {speed:.1f} km/h")
            return event_id
            
        except Exception as e:
            logger.error(f"Failed to log event: {e}")
            return None

    def _save_vehicle_crop(
        self,
        frame: np.ndarray,
        bbox: List[float],
        dt_str: str,
        class_name: str,
        speed: float
    ) -> str:
        """Save cropped vehicle image."""
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in bbox]
        
        # Clamp coordinates
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        if x2 <= x1 or y2 <= y1:
            return ""
        
        crop = frame[y1:y2, x1:x2]
        filename = f"{dt_str}_{class_name}_{speed:.0f}kmh.jpg"
        file_path = os.path.join(self.captures_dir, filename)
        cv2.imwrite(file_path, crop)
        
        return filename

    def _save_plate_crop(
        self,
        plate_crop: Optional[np.ndarray],
        dt_str: str
    ) -> str:
        """Save cropped license plate image."""
        if plate_crop is None or plate_crop.size == 0:
            return ""
        
        filename = f"{dt_str}_plate.jpg"
        file_path = os.path.join(self.captures_dir, filename)
        cv2.imwrite(file_path, plate_crop)
        
        return filename

    def get_events(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """
        Retrieve traffic events from database.
        
        Args:
            limit: Maximum number of events to return
            offset: Number of events to skip
            
        Returns:
            List of event dictionaries
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT * FROM traffic_events 
                    ORDER BY timestamp DESC 
                    LIMIT ? OFFSET ?
                ''', (limit, offset))
                rows = cursor.fetchall()
            
            return [self._row_to_dict(row) for row in rows]
            
        except Exception as e:
            logger.error(f"Failed to get events: {e}")
            return []

    def _row_to_dict(self, row: sqlite3.Row) -> Dict[str, Any]:
        """Convert database row to dictionary with safe encoding."""
        result = {}
        for key in row.keys():
            value = row[key]
            if isinstance(value, bytes):
                try:
                    result[key] = value.decode('utf-8')
                except UnicodeDecodeError:
                    result[key] = value.decode('utf-8', errors='replace')
                    logger.warning(f"Field {key} had encoding issues")
            else:
                result[key] = value
        return result

    def delete_event(self, event_id: int) -> bool:
        """
        Delete a specific event and its associated images.
        
        Args:
            event_id: ID of the event to delete
            
        Returns:
            True if deleted successfully, False otherwise
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Get image paths first
                cursor.execute(
                    'SELECT image_path, plate_image_path FROM traffic_events WHERE id = ?',
                    (event_id,)
                )
                row = cursor.fetchone()
                
                if not row:
                    return False
                
                # Delete images
                for filename in [row['image_path'], row['plate_image_path']]:
                    if filename:
                        full_path = os.path.join(self.captures_dir, filename)
                        if os.path.exists(full_path):
                            try:
                                os.remove(full_path)
                            except OSError as err:
                                logger.warning(f"Failed to remove {full_path}: {err}")
                
                # Delete record
                cursor.execute('DELETE FROM traffic_events WHERE id = ?', (event_id,))
                
            logger.info(f"Deleted event #{event_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete event {event_id}: {e}")
            return False

    def delete_all_events(self) -> bool:
        """
        Delete all events and their associated images.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('DELETE FROM traffic_events')
            
            # Delete all files in captures directory
            self._clear_captures_dir()
            
            logger.info("Deleted all traffic events")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete all events: {e}")
            return False

    def _clear_captures_dir(self) -> None:
        """Remove all files from captures directory."""
        if not os.path.exists(self.captures_dir):
            return
            
        for filename in os.listdir(self.captures_dir):
            file_path = os.path.join(self.captures_dir, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                logger.warning(f"Failed to delete {file_path}: {e}")

    def get_event_count(self) -> int:
        """Get total number of events in database."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT COUNT(*) FROM traffic_events')
                return cursor.fetchone()[0]
        except Exception:
            return 0

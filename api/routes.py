"""API routes for the traffic detection server."""

import os
import shutil
import logging
import cv2
from typing import List

from fastapi import APIRouter, Request, UploadFile, File, Response
from fastapi.responses import JSONResponse

from config import ConfigManager
from db import DatabaseManager
from constants import VIDEO_EXTENSIONS

logger = logging.getLogger(__name__)


def create_config_router(config_manager: ConfigManager) -> APIRouter:
    """Create router for configuration endpoints."""
    router = APIRouter(prefix="/api/config", tags=["config"])
    
    @router.get("/lanes")
    async def get_lanes():
        """Get current lane configurations."""
        return {"lanes": config_manager.get_lanes()}
    
    @router.post("/lanes")
    async def set_lanes(request: Request):
        """
        Set lane configurations.
        
        Expected payload:
        {
            "lanes": [
                {
                    "name": "Northbound",
                    "line_a": [x1, y1, x2, y2],
                    "line_b": [x1, y1, x2, y2],
                    "distance": 5.0
                }
            ]
        }
        """
        data = await request.json()
        lanes = data.get("lanes", [])
        config_manager.set_lanes(lanes)
        
        logger.info(f"Updated {len(lanes)} lane configurations")
        return {"status": "updated", "lane_count": len(lanes)}
    
    @router.get("/zones")
    async def get_zones():
        """Get current zone configurations."""
        return {"zones": config_manager.get_zones()}

    @router.post("/zones")
    async def set_zones(request: Request):
        """
        Set zone configurations.
        
        Expected payload:
        {
            "zones": [
                {
                    "id": "zone-1",
                    "name": "Direction Zone 1",
                    "type": "direction",  // "direction" or "plate"
                    "polygon": [x1, y1, x2, y2, x3, y3, x4, y4]  // flat normalized coords
                }
            ]
        }
        """
        data = await request.json()
        zones = data.get("zones", [])
        config_manager.set_zones(zones)
        
        logger.info(f"Updated {len(zones)} zone configurations")
        return {"status": "updated", "zone_count": len(zones)}
    
    @router.get("/modules")
    async def get_modules():
        """Get current module configuration."""
        return {
            "modules": config_manager.get_modules(),
            "plate_trigger": config_manager.get_plate_trigger(),
            "speed_threshold": config_manager.get_speed_threshold()
        }
    
    @router.post("/modules")
    async def set_modules(request: Request):
        """
        Set module configuration.
        
        Expected payload:
        {
            "modules": {
                "speed": true,
                "direction": true,
                "plate": false
            },
            "plate_trigger": "on_exit",  // optional
            "speed_threshold": 80.0      // optional
        }
        """
        data = await request.json()
        
        if "modules" in data:
            config_manager.set_modules(data["modules"])
        
        if "plate_trigger" in data:
            config_manager.set_plate_trigger(data["plate_trigger"])
        
        if "speed_threshold" in data:
            config_manager.set_speed_threshold(data["speed_threshold"])
        
        logger.info(f"Updated module configuration: {config_manager.get_modules()}")
        return {
            "status": "updated",
            "modules": config_manager.get_modules(),
            "plate_trigger": config_manager.get_plate_trigger(),
            "speed_threshold": config_manager.get_speed_threshold()
        }
    
    @router.get("/lines")
    async def get_lines():
        """Legacy endpoint for backwards compatibility."""
        config_manager.reload()
        return config_manager.config.to_dict()
    
    @router.get("/plate_line")
    async def get_plate_line():
        """Get the plate capture line configuration."""
        return {"plate_line": config_manager.get_plate_line()}
    
    @router.post("/plate_line")
    async def set_plate_line(request: Request):
        """
        Set the plate capture line.
        
        Expected payload:
        {
            "plate_line": [x1, y1, x2, y2]  // normalized 0-1
        }
        """
        data = await request.json()
        line = data.get("plate_line")
        config_manager.set_plate_line(line)
        logger.info(f"Updated plate line: {line}")
        return {"status": "updated", "plate_line": line}
    
    return router


def create_events_router(db_manager: DatabaseManager) -> APIRouter:
    """Create router for event management endpoints."""
    router = APIRouter(prefix="/api", tags=["events"])
    
    @router.get("/events")
    async def get_events(limit: int = 100, offset: int = 0):
        """Retrieve traffic events."""
        events = db_manager.get_events(limit=limit, offset=offset)
        return JSONResponse(content=events)
    
    @router.delete("/events/{event_id}")
    async def delete_event(event_id: int):
        """Delete a specific event."""
        success = db_manager.delete_event(event_id)
        if success:
            return {"status": "deleted", "id": event_id}
        return JSONResponse(
            status_code=404, 
            content={"error": "Event not found"}
        )
    
    @router.delete("/events")
    async def delete_all_events():
        """Delete all events."""
        db_manager.delete_all_events()
        return {"status": "deleted", "count": "all"}
    
    return router


def create_video_router(videos_dir: str) -> APIRouter:
    """Create router for video management endpoints."""
    router = APIRouter(prefix="/api", tags=["videos"])
    
    @router.get("/videos")
    async def list_videos():
        """List available video files."""
        if not os.path.exists(videos_dir):
            return {"videos": []}
        
        videos = [
            f for f in os.listdir(videos_dir) 
            if f.lower().endswith(VIDEO_EXTENSIONS)
        ]
        return {"videos": sorted(videos)}
    
    @router.post("/videos/upload")
    async def upload_video(file: UploadFile = File(...)):
        """Upload a new video file."""
        if not os.path.exists(videos_dir):
            os.makedirs(videos_dir)
        
        file_path = os.path.join(videos_dir, file.filename)
        
        try:
            with open(file_path, "wb") as f:
                while chunk := await file.read(1024 * 1024):  # 1MB chunks
                    f.write(chunk)
            
            logger.info(f"Uploaded video: {file.filename}")
            return {"status": "uploaded", "filename": file.filename}
        except Exception as e:
            logger.error(f"Upload failed: {e}")
            return JSONResponse(
                status_code=500,
                content={"error": str(e)}
            )
    
    @router.delete("/videos/{filename}")
    async def delete_video(filename: str):
        """Delete a video file."""
        file_path = os.path.join(videos_dir, filename)
        
        if os.path.exists(file_path):
            os.remove(file_path)
            return {"status": "deleted", "filename": filename}
        
        return JSONResponse(
            status_code=404,
            content={"error": "Video not found"}
        )
    
    @router.get("/video_preview")
    async def video_preview(video_source: str):
        """Get a preview frame from a video."""
        if not video_source:
            return JSONResponse(
                status_code=400,
                content={"error": "video_source required"}
            )
        
        video_path = os.path.join(videos_dir, video_source)
        
        if not os.path.exists(video_path):
            return JSONResponse(
                status_code=404,
                content={"error": "Video not found"}
            )
        
        try:
            cap = cv2.VideoCapture(video_path)
            ret, frame = cap.read()
            cap.release()
            
            if not ret:
                return JSONResponse(
                    status_code=500,
                    content={"error": "Could not read frame"}
                )
            
            _, buffer = cv2.imencode('.jpg', frame)
            return Response(
                content=buffer.tobytes(),
                media_type="image/jpeg"
            )
        except Exception as e:
            logger.error(f"Preview failed: {e}")
            return JSONResponse(
                status_code=500,
                content={"error": str(e)}
            )
    
    return router

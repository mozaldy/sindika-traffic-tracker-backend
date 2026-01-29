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
    
    @router.get("/lines")
    async def get_lines():
        """Legacy endpoint for backwards compatibility."""
        config_manager.reload()
        return config_manager.config.to_dict()
    
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

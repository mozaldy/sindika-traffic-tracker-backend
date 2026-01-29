"""
Traffic Detection Server
========================

FastAPI application for real-time traffic analysis via WebRTC streaming.
Provides video processing, object detection, speed estimation, and event logging.
"""

import os
import asyncio
import logging
from typing import Set

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaRelay
import uvicorn

from api import (
    TrafficAnalysisTrack,
    create_config_router,
    create_events_router,
    create_video_router
)
from config import ConfigManager
from db import DatabaseManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("traffic_server")

# Paths
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(ROOT_DIR, "config.json")
VIDEOS_DIR = os.path.join(ROOT_DIR, "videos")
CAPTURES_DIR = os.path.join(ROOT_DIR, "captures")


class TrafficServer:
    """
    Main traffic detection server application.
    
    Manages FastAPI app, WebRTC connections, and shared resources.
    """
    
    def __init__(self):
        """Initialize the server with all components."""
        # Shared resources
        self.config_manager = ConfigManager(CONFIG_FILE)
        self.db_manager = DatabaseManager(captures_dir=CAPTURES_DIR)
        self.relay = MediaRelay()
        self.peer_connections: Set[RTCPeerConnection] = set()
        
        # Create FastAPI app
        self.app = self._create_app()
        
        logger.info("TrafficServer initialized")
    
    def _create_app(self) -> FastAPI:
        """Create and configure FastAPI application."""
        app = FastAPI(
            title="Traffic Detection API",
            description="Real-time traffic analysis with WebRTC streaming",
            version="2.0.0"
        )
        
        # CORS middleware
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Static files
        self._setup_static_files(app)
        
        # API routers
        app.include_router(create_config_router(self.config_manager))
        app.include_router(create_events_router(self.db_manager))
        app.include_router(create_video_router(VIDEOS_DIR))
        
        # WebRTC endpoint
        self._setup_webrtc_endpoint(app)
        
        # Control endpoints
        self._setup_control_endpoints(app)
        
        # Lifecycle events
        app.on_event("shutdown")(self._on_shutdown)
        
        return app
    
    def _setup_static_files(self, app: FastAPI) -> None:
        """Configure static file serving."""
        # Ensure directories exist
        for dir_path in [VIDEOS_DIR, CAPTURES_DIR]:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
        
        # Mount static directories
        app.mount("/videos", StaticFiles(directory=VIDEOS_DIR), name="videos")
        app.mount("/captures", StaticFiles(directory=CAPTURES_DIR), name="captures")
        
    def _setup_control_endpoints(self, app: FastAPI) -> None:
        """Configure control endpoints."""
        
        @app.post("/api/control/pause")
        async def pause_stream():
            """Pause all active streams."""
            count = 0
            for pc in self.peer_connections:
                if "video_track" in pc.__dict__:
                    pc.__dict__["video_track"].pause()
                    count += 1
            return {"status": "paused", "count": count}
            
        @app.post("/api/control/resume")
        async def resume_stream():
            """Resume all active streams."""
            count = 0
            for pc in self.peer_connections:
                if "video_track" in pc.__dict__:
                    pc.__dict__["video_track"].resume()
                    count += 1
            return {"status": "resumed", "count": count}
    
    def _setup_webrtc_endpoint(self, app: FastAPI) -> None:
        """Configure WebRTC offer/answer endpoint."""
        
        @app.post("/api/offer")
        async def offer(request: Request):
            """Handle WebRTC offer and return answer."""
            return await self._handle_offer(request)
    
    async def _handle_offer(self, request: Request) -> JSONResponse:
        """
        Process WebRTC offer and establish connection.
        
        Creates a new TrafficAnalysisTrack for the requested video source
        and sets up the peer connection.
        """
        params = await request.json()
        
        offer = RTCSessionDescription(
            sdp=params["sdp"], 
            type=params["type"]
        )
        video_source = params.get("video_source", "")
        target_classes = params.get("target_classes")
        
        # Validate video source
        video_path = os.path.join(VIDEOS_DIR, video_source)
        if not os.path.exists(video_path):
            return JSONResponse(
                status_code=400,
                content={"error": f"Video not found: {video_source}"}
            )
        
        # Create peer connection
        pc = RTCPeerConnection()
        self.peer_connections.add(pc)
        
        # Create and add video track
        try:
            video_track = TrafficAnalysisTrack(
                source_path=video_path,
                config_manager=self.config_manager,
                target_classes=target_classes
            )
            # Attach track to pc for cleanup
            pc.__dict__["video_track"] = video_track
            
            pc.addTrack(self.relay.subscribe(video_track))
        except Exception as e:
            logger.error(f"Failed to create video track: {e}")
            await pc.close()
            self.peer_connections.discard(pc)
            return JSONResponse(
                status_code=500,
                content={"error": str(e)}
            )
        
        @pc.on("connectionstatechange")
        async def on_connection_state_change():
            logger.info(f"Connection state: {pc.connectionState}")
            if pc.connectionState in ["failed", "closed", "disconnected"]:
                # Stop the track if it exists
                if "video_track" in pc.__dict__:
                    logger.info("Stopping video track")
                    pc.__dict__["video_track"].stop()
                
                await pc.close()
                self.peer_connections.discard(pc)
        
        # Handle offer/answer
        await pc.setRemoteDescription(offer)
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        
        return JSONResponse(content={
            "sdp": pc.localDescription.sdp,
            "type": pc.localDescription.type
        })
    
    async def _on_shutdown(self) -> None:
        """Clean up resources on server shutdown."""
        logger.info("Shutting down... closing peer connections")
        
        close_tasks = [pc.close() for pc in self.peer_connections]
        await asyncio.gather(*close_tasks)
        self.peer_connections.clear()
        
        logger.info("Shutdown complete")


# Create server instance
server = TrafficServer()
app = server.app


def main():
    """Run the server."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Traffic Detection Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    args = parser.parse_args()
    
    logger.info(f"Starting server on {args.host}:{args.port}")
    
    uvicorn.run(
        "main:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )


if __name__ == "__main__":
    main()

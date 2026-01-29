"""
Traffic Detection Server
========================

FastAPI application for real-time traffic analysis.
This server integrates the modular backend components:
- API Routes (Config, Videos, Events)
- WebRTC Streaming (TrafficAnalysisTrack)
- Logic Engine (Turn, Speed, Plate Detection)
"""

import os
import json
import logging
import asyncio
from typing import Dict, Set

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from aiortc import RTCPeerConnection, RTCSessionDescription

from config import ConfigManager
from api.routes import create_config_router, create_video_router, create_events_router
from api.streaming import TrafficAnalysisTrack
from db import DatabaseManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("server")

# Paths
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(ROOT_DIR, "config.json")
VIDEOS_DIR = os.path.join(ROOT_DIR, "videos")
CAPTURES_DIR = os.path.join(ROOT_DIR, "captures")

# Ensure directories exist
os.makedirs(VIDEOS_DIR, exist_ok=True)
os.makedirs(CAPTURES_DIR, exist_ok=True)

# Initialize resources
config_manager = ConfigManager(CONFIG_FILE)
db_manager = DatabaseManager()
pcs: Set[RTCPeerConnection] = set()

# Initialize API routers
config_router = create_config_router(config_manager)
video_router = create_video_router(VIDEOS_DIR)
event_router = create_events_router(db_manager)

# Setup FastAPI
app = FastAPI(title="Traffic Detection Server")

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Routers
app.include_router(config_router)
app.include_router(video_router)
app.include_router(event_router)

# Mount Static Files (Frontend)
static_dir = os.path.join(ROOT_DIR, "static")
if not os.path.exists(static_dir):
    os.makedirs(static_dir)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/")
def read_root():
    return {"status": "online", "service": "traffic-tracker"}

@app.post("/api/offer")
async def offer(request: Request):
    """WebRTC offer endpoint."""
    params = await request.json()
    off = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
    
    # Get configuration from request
    video_source = params.get("video_source", "test_video.mp4")
    video_path = os.path.join(VIDEOS_DIR, video_source)
    
    pc = RTCPeerConnection()
    pcs.add(pc)
    
    # Determine target classes (if any)
    target_classes = params.get("target_classes")
    
    # Create Analysis Track
    track = TrafficAnalysisTrack(
        source_path=video_path,
        config_manager=config_manager,
        target_classes=target_classes
    )
    
    pc.addTrack(track)
    
    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        logger.info(f"Connection state is {pc.connectionState}")
        if pc.connectionState == "failed":
            await pc.close()
            pcs.discard(pc)
            track.stop()
        elif pc.connectionState == "closed":
            pcs.discard(pc)
            track.stop()

    await pc.setRemoteDescription(off)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)
    
    return {
        "sdp": pc.localDescription.sdp,
        "type": pc.localDescription.type
    }

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Server shutting down, closing connections...")
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

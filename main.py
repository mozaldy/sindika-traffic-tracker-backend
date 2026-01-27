
import argparse
import asyncio
import json
import logging
import os
import cv2
import platform
from typing import List, Optional


from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from aiortc.contrib.media import MediaRelay
from av import VideoFrame

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Import our custom CV engine
from traffic_engine import ObjectDetector, ObjectTracker, StatsManager, TrafficVisualizer, SpeedEstimator, COCO_CLASSES

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("traffic_server")

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

class TrafficAnalysisTrack(VideoStreamTrack):
    """
    A WebRTC VideoStreamTrack that reads frames from a source (file/camera),
    runs the traffic analysis pipeline, and yields the annotated frame.
    """
    def __init__(self, source_path: str, target_classes: List[str] = None):
        super().__init__()
        self.source_path = source_path
        
        # Initialize CV Components
        logger.info(f"Initializing Traffic Analysis Engine for source: {source_path}")
        
        # Map target classes to IDs
        target_ids = None
        if target_classes:
            target_ids = []
            name_to_id = {v: k for k, v in COCO_CLASSES.items()}
            for name in target_classes:
                if name in name_to_id:
                    target_ids.append(name_to_id[name])
                else:
                    logger.warning(f"Class '{name}' not found.")
        
        self.detector = ObjectDetector(confidence_threshold=0.5, target_classes=target_ids)
        self.tracker = ObjectTracker()
        self.stats = StatsManager()
        self.visualizer = TrafficVisualizer()
        self.speed_estimator = SpeedEstimator()
        
        # Video Capture
        self.cap = cv2.VideoCapture(self.source_path)
        if not self.cap.isOpened():
            logger.error(f"Failed to open video source: {self.source_path}")
            
        self.loop = asyncio.get_event_loop()
        
    def _process_frame(self, frame):
        """
        Sync function to run the CV pipeline.
        """
        # Run Analysis Pipeline
        # 1. Detect
        detections = self.detector.detect(frame)
        
        # 2. Track
        detections = self.tracker.update(detections)
        
        # 3. Speed
        self.speed_estimator.update(detections)
        
        # 4. Stats
        self.stats.update(detections)
        
        # 5. Annotate
        annotated_frame = self.visualizer.annotate(frame, detections, self.stats, self.speed_estimator)
        
        return annotated_frame

    async def recv(self):
        """
        The core loop. Called when the client requests a frame.
        """
        pts, time_base = await self.next_timestamp()
        
        # Read frame from OpenCV
        ret, frame = self.cap.read()
        if not ret:
            # Loop video
            logger.info("Video finished, looping...")
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()
            if not ret:
                # If still failing, return a black frame or error
                logger.error("Could not read frame even after loop.")
                return None # This might close the track

        # Run CV Pipeline in a separate thread to ensure we don't block the AsyncIO loop (which handles WebRTC heartbeats)
        annotated_frame = await self.loop.run_in_executor(None, self._process_frame, frame)
        
        # Convert to WebRTC Frame (AV)
        # OpenCV is BGR, av.VideoFrame.from_ndarray expects BGR if format='bgr24'
        new_frame = VideoFrame.from_ndarray(annotated_frame, format="bgr24")
        new_frame.pts = pts
        new_frame.time_base = time_base
        
        return new_frame

    def stop(self):
        if self.cap.isOpened():
            self.cap.release()
        super().stop()


# --- FastAPI App ---
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files (for index.html)
if not os.path.exists(os.path.join(ROOT_DIR, "static")):
    os.makedirs(os.path.join(ROOT_DIR, "static"))
    
app.mount("/static", StaticFiles(directory=os.path.join(ROOT_DIR, "static")), name="static")

# Store active peer connections to close them on shutdown
pcs = set()

@app.get("/", response_class=HTMLResponse)
async def read_root():
    return open(os.path.join(ROOT_DIR, "static/index.html")).read()

@app.post("/api/offer")
async def offer(request: Request):
    params = await request.json()
    sdp = params["sdp"]
    type_ = params["type"]
    
    offer = RTCSessionDescription(sdp=sdp, type=type_)
    
    pc = RTCPeerConnection()
    pcs.add(pc)
    
    logger.info("Created new PeerConnection")
    
    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        logger.info(f"Connection state is {pc.connectionState}")
        if pc.connectionState == "failed":
            await pc.close()
            pcs.discard(pc)
    
    # Create our custom video track
    # NOTE: In a real app, source might be dynamic
    video_source = "./traffic_fixed.mp4"
    if not os.path.exists(video_source):
        # Fallback for testing if file moved
        video_source = "../traffic.mp4"
        
    target_classes = ["person", "car", "motorcycle", "truck"]
    
    video_track = TrafficAnalysisTrack(video_source, target_classes)
    
    pc.addTrack(video_track)

    await pc.setRemoteDescription(offer)
    
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)
    
    return JSONResponse(content={
        "sdp": pc.localDescription.sdp,
        "type": pc.localDescription.type
    })

@app.on_event("shutdown")
async def on_shutdown():
    logger.info("Shutting down... closing peer connections.")
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)


import argparse
import asyncio
import json
import logging
import os
import cv2
import platform
import shutil
from typing import List, Optional


from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from aiortc.contrib.media import MediaRelay
from av import VideoFrame

from fastapi import FastAPI, Request, File, UploadFile, Response
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Import our custom CV engine
from traffic_engine import ObjectDetector, ObjectTracker, StatsManager, TrafficVisualizer, PolygonZoneEstimator, COCO_CLASSES
from db_manager import DatabaseManager

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("traffic_server")

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Config Persistence ---
CONFIG_FILE = os.path.join(ROOT_DIR, "config.json")

def load_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_config(config):
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f)

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
        self.speed_estimator = PolygonZoneEstimator()
        
        # Load saved config
        config = load_config()
        if "polygon" in config:
             self.speed_estimator.set_config(
                config["polygon"],
                config.get("distance", 5.0)
            )
        elif "line1" in config and "line2" in config:
            self.speed_estimator.set_config_lines(
                config["line1"],
                config["line2"],
                config.get("distance", 5.0)
            )

        self.db = DatabaseManager()
        
        # Video Capture
        self.cap = cv2.VideoCapture(self.source_path)
        if not self.cap.isOpened():
            logger.error(f"Failed to open video source: {self.source_path}")
            
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration_sec = self.total_frames / self.fps if self.fps > 0 else 0

        self.loop = asyncio.get_event_loop()
        
    def _format_time(self, seconds):
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins:02d}:{secs:02d}"

    def _process_frame(self, frame, timestamp_info, timestamp_sec):
        """
        Sync function to run the CV pipeline.
        """
        # Run Analysis Pipeline
        # 1. Detect
        detections = self.detector.detect(frame)
        
        # 2. Track
        detections = self.tracker.update(detections)
        
        # 3. Speed
        self.speed_estimator.update(detections, frame.shape, timestamp_sec)
        
        # 4. Check for completed events and Log
        # We iterate over detections to match ID with completed events
        if detections.tracker_id is not None:
             for xyxy, class_id, tracker_id in zip(detections.xyxy, detections.class_id, detections.tracker_id):
                 if tracker_id in self.speed_estimator.completed_speeds:
                     event = self.speed_estimator.completed_speeds[tracker_id]
                     
                     if not event.get("logged", False):
                         class_name = COCO_CLASSES[class_id]
                         self.db.log_event(
                             frame=frame,
                             bbox=xyxy,
                             class_name=class_name,
                             speed=event['speed'],
                             direction=event['direction'], # Relative Angle
                             direction_symbol=event.get('direction_symbol'),
                             video_source=os.path.basename(self.source_path),
                             crossing_start=event.get('start_time', 0.0),
                             crossing_end=event.get('end_time', 0.0)
                         )
                         event["logged"] = True

        # 5. Stats
        self.stats.update(detections)
        
        # 6. Annotate
        annotated_frame = self.visualizer.annotate(frame, detections, self.stats, self.speed_estimator, timestamp_info)
        
        return annotated_frame

    async def recv(self):
        """
        The core loop. Called when the client requests a frame.
        """
        pts, time_base = await self.next_timestamp()
        
        # Read frame from OpenCV
        ret, frame = self.cap.read()
        if not ret:
            # Loop video: Re-open the file to handle different codecs/containers
            logger.info("Video finished, looping (re-opening file)...")
            self.cap.release()
            
            self.cap = cv2.VideoCapture(self.source_path)
            
            # Additional check if re-open succeeded
            if not self.cap.isOpened():
                logger.error(f"Failed to re-open video source: {self.source_path}")
                return None

            ret, frame = self.cap.read()
            if not ret:
                # If still failing, return a black frame or error
                logger.error("Could not read frame even after re-opening.")
                return None # This might close the track

        # Calculate Timestamp Info
        current_frame_pos = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
        current_sec = current_frame_pos / self.fps if self.fps > 0 else 0
        timestamp_str = f"{self._format_time(current_sec)} / {self._format_time(self.duration_sec)}"

        # Run CV Pipeline in a separate thread to ensure we don't block the AsyncIO loop (which handles WebRTC heartbeats)
        annotated_frame = await self.loop.run_in_executor(None, self._process_frame, frame, timestamp_str, current_sec)
        
        if annotated_frame is None: 
             return None

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

# --- Video Management ---
VIDEO_DIR = os.path.join(ROOT_DIR, "videos")
if not os.path.exists(VIDEO_DIR):
    os.makedirs(VIDEO_DIR)

@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)):
    try:
        file_path = os.path.join(VIDEO_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info(f"Uploaded video: {file.filename}")
        return {"filename": file.filename, "status": "success"}
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        return JSONResponse(status_code=500, content={"message": str(e)})

@app.get("/api/videos")
async def list_videos():
    files = []
        
    if os.path.exists(VIDEO_DIR):
        for f in os.listdir(VIDEO_DIR):
            if f.endswith(('.mp4', '.avi', '.mov')):
                files.append(f)
    return {"videos": sorted(list(set(files)))}

@app.get("/api/video_preview")
async def video_preview(video_source: str):
    if not video_source:
        return Response(status_code=400)
        
    # Resolve path consistent with other endpoints
    # 1. Check if absolute or direct relative path exists
    if os.path.exists(video_source):
        path = video_source
    else:
        # 2. Check in VIDEO_DIR
        path = os.path.join(VIDEO_DIR, video_source)
        if not os.path.exists(path):
             return Response(status_code=404, content="Video not found")

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
         return Response(status_code=500, content="Could not open video")
    
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        success, buffer = cv2.imencode('.jpg', frame)
        if success:
            return Response(content=buffer.tobytes(), media_type="image/jpeg")
            
    return Response(status_code=500, content="Could not read frame")

@app.post("/api/offer")
async def offer(request: Request):
    params = await request.json()
    sdp = params["sdp"]
    type_ = params["type"]
    video_source_name = params.get("video_source")
    target_classes = params.get("target_classes", ["person", "car", "motorcycle", "truck", "bus"])

    offer = RTCSessionDescription(sdp=sdp, type=type_)
    
    pc = RTCPeerConnection()
    pcs.add(pc)
    
    logger.info(f"Created new PeerConnection. Selected Source: {video_source_name}, Classes: {target_classes}")
    
    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        logger.info(f"Connection state is {pc.connectionState}")
        if pc.connectionState == "failed":
            await pc.close()
            pcs.discard(pc)
    
    # Determine absolute path for video source
    if not video_source_name:
        return JSONResponse(status_code=400, content={"message": "No video source selected. Please upload a video."})

    video_path = os.path.join(VIDEO_DIR, video_source_name)
        
    if not os.path.exists(video_path):
         logger.error(f"Source {video_path} not found.")
         return JSONResponse(status_code=404, content={"message": f"Video '{video_source_name}' not found. Please upload it."})
    
    video_track = TrafficAnalysisTrack(video_path, target_classes)
    
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

# --- Config Endpoints ---

@app.get("/api/config/lines")
async def get_config_lines():
    config = load_config()
    return config

@app.post("/api/config/lines")
async def config_lines(request: Request):
    data = await request.json()
    
    # Save to disk (Persistent)
    save_config(data)
    
    polygon = data.get("polygon")
   
    distance = data.get("distance", 5.0)
    
    # Update all active tracks
    for pc in pcs:
        for transceiver in pc.getTransceivers():
             if transceiver.sender.track and isinstance(transceiver.sender.track, TrafficAnalysisTrack):
                 track = transceiver.sender.track
                 if polygon:
                    # Generic polygon update
                    track.speed_estimator.set_config(polygon, distance)
                 else:
                     # Fallback line update (which maps to polygon)
                     line1 = data.get("line1")
                     line2 = data.get("line2")
                     if line1 and line2:
                         track.speed_estimator.set_config_lines(line1, line2, distance)
                         
                 logger.info("Updated speed config for active track via config/lines.")
                 
    return {"status": "updated"}

# Support the config/zone endpoint (Friend's API) but link it to persistence if desired
# Or just keep it ephemeral? User's legacy code used persistence.
# Let's make this endpoint ALSO save if we want consistency, but friend's code didn't seem to have save logic.
# I'll just make it update the active track, but maybe NOT save to `lines` config logic to avoid format clashes unless we unify.
# User's frontend calls config/lines?
# If I look at the diffs, `config_zone` endpoint was Friend's work.
@app.post("/api/config/zone")
async def config_zone(request: Request):
    data = await request.json()
    # Support both keys for robustness
    zone = data.get("zone") or data.get("polygon") 
    distance = data.get("distance", 5.0)
    
    # Try to save this as 'polygon' in the config for persistence
    current_config = load_config()
    current_config["polygon"] = zone
    current_config["distance"] = distance
    save_config(current_config)

    for pc in pcs:
        for transceiver in pc.getTransceivers():
             if transceiver.sender.track and isinstance(transceiver.sender.track, TrafficAnalysisTrack):
                 track = transceiver.sender.track
                 if zone:
                     track.speed_estimator.set_config(zone, distance)
                     logger.info("Updated zone config for active track via config/zone.")
                 
    return {"status": "updated"}

# --- Event Management Endpoints ---
@app.get("/api/events")
async def get_events(limit: int = 100, offset: int = 0):
    db = DatabaseManager()
    events = db.get_events(limit, offset)
    return {"events": events}

@app.delete("/api/events/{event_id}")
async def delete_event(event_id: int):
    db = DatabaseManager()
    success = db.delete_event(event_id)
    if success:
        return {"status": "deleted", "id": event_id}
    return JSONResponse(status_code=404, content={"message": "Event not found"})

@app.delete("/api/events")
async def delete_all_events():
    db = DatabaseManager()
    success = db.delete_all_events()
    if success:
        return {"status": "cleared"}
    return JSONResponse(status_code=500, content={"message": "Failed to clear events"})

# Mount captures directory to serve images
if not os.path.exists(os.path.join(ROOT_DIR, "captures")):
    os.makedirs(os.path.join(ROOT_DIR, "captures"))
app.mount("/captures", StaticFiles(directory=os.path.join(ROOT_DIR, "captures")), name="captures")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

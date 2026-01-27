# Sindika Traffic Tracker Backend

This is the FastAPI backend for the Sindika Traffic Tracker. It handles real-time object detection and tracking on video streams using RF-DETR and WebRTC.

## Prerequisites

-   Python 3.10+
-   CUDA-enabled GPU (Recommended for RF-DETR)

## Installation

1.  **Clone the repository**:
    ```bash
    git clone <your-repo-url>
    cd sindika-traffic-tracker-backend
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Setup (Crucial!)

### 1. Model File
The **RF-DETR Medium** model checkpoint (`rf-detr-medium.pth`) is **NOT** included in this repository due to its large size (~387MB).

You must download/place the model file in the root of this backend directory:
`./rf-detr-medium.pth`

*(If the model is not found, the application might attempt to download it automatically depending on the `rfdetr` library configuration, or crash)*.

### 2. Video Source
By default, the backend looks for a video file named `traffic_fixed.mp4` in the root directory for testing:
`./traffic_fixed.mp4`

If you want to use a different source (like a webcam or RTSP stream), you must modify `main.py`:
```python
# main.py
video_source = 0  # For Webcam
# OR
video_source = "rtsp://..."  # For IP Camera
```

## Running the Server

Start the FastAPI server using Uvicorn:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The server will start on `http://localhost:8000`.

## API Endpoints

-   `POST /api/offer`: WebRTC signaling endpoint. Receives an SDP offer and returns an answer.

## Architecture

-   `main.py`: FastAPI server logic and WebRTC track creation.
-   `traffic_engine.py`: Core Computer Vision pipeline (Detection, Tracking, Speed Estimation).

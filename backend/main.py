import shutil
from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import os
import sys
from dotenv import load_dotenv
import base64
import numpy as np
import cv2
import json
from typing import Optional
import time # Import time module

# Load environment variables from .env file in the root directory
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path=dotenv_path)

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from accident_detection import AccidentDetector

# Construct the absolute path to the model file
model_path = os.path.join(os.path.dirname(__file__), '..', 'best.pt')

app = FastAPI()

# CORS configuration
origins = [
    "http://localhost:3000",
    "http://localhost:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.websocket("/ws/live-detection")
async def websocket_endpoint(websocket: WebSocket, lang: Optional[str] = 'en'):
    await websocket.accept()
    # Initialize detector with the language from the websocket connection
    detector = AccidentDetector(model_path=model_path, language=lang) 
    try:
        while True:
            data = await websocket.receive_text()
            if data.startswith('data:image/jpeg;base64,'):
                header, encoded = data.split(',', 1)
                frame_data = base64.b64decode(encoded)
                frame = cv2.imdecode(np.frombuffer(frame_data, np.uint8), cv2.IMREAD_COLOR)

                # Process the frame
                result = detector.process_frame(frame)
                
                await websocket.send_text(json.dumps(result))

    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        await websocket.close()


@app.post("/upload-video")
async def upload_video(video: UploadFile = File(...), lang: Optional[str] = 'en'):
    """
    Receives a video file, saves it to the 'uploads' directory, and
    triggers the accident detection script.
    """
    upload_dir = "uploads"
    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir)

    file_path = os.path.join(upload_dir, video.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(video.file, buffer)

    print(f"Video '{video.filename}' uploaded. Running accident detection...")
    
    # Initialize detector with the language from the upload request
    detector = AccidentDetector(model_path=model_path, language=lang)
    # process_video now returns the actual report object, not just a filename
    report_object = detector.process_video(file_path) 

    if report_object:
        # Save the report with a unique filename and return that filename
        report_filename = f"accident_report_{int(time.time())}.json"
        report_path = os.path.join(detector.accident_report_dir, report_filename)
        with open(report_path, "w") as f:
            json.dump(report_object, f, indent=4)
        return {"report_filename": report_filename}
    else:
        return {"error": "Accident detection failed."}

@app.get("/reports/{filename}")
async def get_report(filename: str):
    """
    Serves the accident report JSON file.
    """
    report_path = os.path.join("accident_reports", filename) # Assuming accident_reports is the directory
    if os.path.exists(report_path):
        return FileResponse(report_path)
    return {"error": "Report not found."}

@app.get("/")
def read_root():
    return {"Hello": "World"}
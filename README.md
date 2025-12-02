# Accident Detection with YOLOv8 and Gemini Pro

This project provides a real-time accident detection system using the YOLOv8 object detection model and Google's Gemini Pro for detailed accident analysis and reporting.

## Features

- **Real-time Accident Detection:** Processes video streams from files or webcams to detect accidents in real-time.
- **Advanced Detection Heuristics:** Utilizes a combination of three methods to ensure high accuracy and reduce false positives:
    1.  **Direct Classification:** A custom-trained YOLOv8 model identifies "vehicle_incident" classes.
    2.  **Collision Detection:** Monitors the Intersection over Union (IoU) of tracked objects to detect collisions.
    3.  **Sudden Stop Analysis:** Tracks object movement to identify sudden and unexpected stops indicative of an accident.
- **AI-Powered Accident Analysis:** Upon confirming an accident, the system sends key video frames to the **Google Gemini 2.5 Pro** model. Gemini then generates a structured JSON report detailing:
    -   **Accident Severity:** (e.g., 'Minor', 'Moderate', 'Severe').
    -   **Concise Description:** A one-sentence summary of the event.
    -   **Inferred Sequence of Events:** A likely timeline of the accident.
    -   **Participants:** Details of involved vehicles or persons (type, color, visible damage, and role).
- **GPS Stamping:** Automatically captures and includes GPS coordinates in the accident report, using location name geocoding or IP-based location as a fallback.
- **Comprehensive Reporting:** Generates a JSON file for each confirmed accident containing:
    -   Timestamp.
    -   GPS coordinates with a Google Maps link.
    -   The detailed AI-generated analysis from Gemini.
    -   The detection parameters used.
- **Annotated Video Output:** Creates a processed video file showing bounding boxes, object tracking paths, and on-screen alerts for potential and confirmed accidents.
- **Highly Configurable:** All major parameters, such as model paths, confidence thresholds, and IoU thresholds, can be adjusted via command-line arguments.

## How It Works

1.  **Video Input:** The system captures frames from a specified video source.
2.  **Object Tracking:** The YOLOv8 model processes each frame to detect and track objects, assigning a unique ID to each.
3.  **Heuristic Analysis:** The system analyzes the tracking data for signs of an accident using the three detection methods.
4.  **Event Confirmation:** An accident is only confirmed if the indicators persist across a configurable number of consecutive frames, preventing false alarms.
5.  **AI Description Generation:** Once confirmed, a buffer of recent frames is sent to the Gemini 2.5 Pro API for analysis.
6.  **Report Generation:** A final report, including the AI analysis and GPS data, is saved to the `accident_reports` directory.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd accident-detection
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up your Gemini API Key:**
    -   Get a free API key from the [Google AI Studio](https://makersuite.google.com/app/apikey).
    -   Create a file named `.env` in the root of the project.
    -   Add your API key to the `.env` file like this:
        ```
        GEMINI_API_KEY='YOUR_API_KEY_HERE'
        ```

## Usage

You can run the accident detection script from the command line.

### Basic Usage

-   **To run on a demo video file:**
    ```bash
    python accident_detection.py --source demo_video.mp4
    ```

-   **To use your webcam:**
    ```bash
    python accident_detection.py --source 0
    ```

### Command-Line Arguments

-   `--source`: Path to the video file or '0' for webcam. (Default: '0')
-   `--model`: Path to the YOLOv8 model file. (Default: 'best.pt')
-   `--output`: Name of the output video file. (Default: 'results_video.mp4')
-   `--acc-conf`: Confidence threshold for 'vehicle_incident' classification. (Default: 0.75)
-   `--frame-threshold`: Number of consecutive frames needed to confirm an accident. (Default: 3)
-   `--speed-threshold`: Pixel displacement threshold to detect a sudden stop. (Default: 10)
-   `--iou-threshold`: IoU threshold for detecting a collision between objects. (Default: 0.1)
-   `--location`: Provide a specific location name (e.g., "Eiffel Tower, Paris") for accurate GPS coordinates in the report.

### Example with Custom Parameters

```bash
python accident_detection.py \
    --source "path/to/your/video.mp4" \
    --model "best.pt" \
    --output "my_analysis_video.mp4" \
    --acc-conf 0.80 \
    --frame-threshold 4 \
    --location "1600 Amphitheatre Parkway, Mountain View, CA"
```

After processing, the output video will be saved as `my_analysis_video.mp4`, and any confirmed accident reports will be stored in the `accident_reports/` directory.

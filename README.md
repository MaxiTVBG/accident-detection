<div align="center">

# 🚗 Accident Detection & AI Analysis 📸

**A cutting-edge system that not only detects traffic accidents in real-time but also uses Generative AI to understand and report on them.**

</div>

<p align="center">
  <img alt="Python Version" src="https://img.shields.io/badge/python-3.9%2B-blue.svg?style=for-the-badge&logo=python">
  <img alt="Frameworks" src="https://img.shields.io/badge/Made%20with-YOLOv8%20%26%20Gemini-orange.svg?style=for-the-badge">
  <img alt="License" src="https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge">
</p>

---

### ✨ Key Features

-   **👁️ Real-time Detection:** Process live video from webcams or files to catch incidents as they happen.
-   **🎯 High-Accuracy Heuristics:** Goes beyond simple detection by using a powerful three-factor confirmation system:
    1.  **AI Classification:** A custom-trained YOLOv8 model spots `vehicle_incident` events.
    2.  **Collision Analysis:** Smartly detects overlapping objects (high IoU) to identify potential collisions.
    3.  **Sudden Stop Monitoring:** Tracks vehicle speeds to flag unnatural, sudden stops.
-   **🧠 Gemini-Powered Insights:** When an accident is confirmed, key frames are sent to **Google's Gemini 2.5 Pro** model, which returns a detailed JSON analysis covering:
    -   **Severity Assessment:** `Minor`, `Moderate`, or `Severe`.
    -   **Event Summary:** A quick, one-sentence description.
    -   **Sequence of Events:** A step-by-step breakdown of the incident.
    -   **Participant Details:** Information on vehicles involved.
-   **🌍 GPS Stamping:** Automatically logs the exact location of the incident, providing a Google Maps link in the final report.
-   **📝 Comprehensive Reports:** Generates a clean `.json` report for each accident, perfect for logging or further analysis.
-   **📹 Annotated Video Output:** Creates a new video file with all detections, tracking paths, and alerts drawn directly onto the footage.

---

### ⚙️ How It Works

1.  **Ingest:** The system reads a video stream frame by frame.
2.  **Detect & Track:** YOLOv8 identifies and puts a tracking ID on every object of interest.
3.  **Analyze:** The system's brain checks for collisions, sudden stops, or direct "incident" classifications.
4.  **Confirm:** To avoid false alarms, an accident is only flagged if the signs persist for several consecutive frames.
5.  **Describe:** Once confirmed, the magic happens. Key frames are sent to the Gemini API.
6.  **Report:** A detailed JSON report is generated and saved in the `accident_reports/` folder.

---

### 🛠️ Getting Started

Follow these steps to get the project running on your local machine.

#### 1. Clone the Repository

```bash
git clone <repository-url>
cd accident-detection
```

#### 2. Set Up a Virtual Environment

```bash
# Create and activate a Python virtual environment
python3 -m venv .venv
source .venv/bin/activate
```

#### 3. Install Dependencies

```bash
# Install all the necessary packages
pip install -r requirements.txt
```

#### 4. Configure Your API Key

> **Note:** This project relies on the Google Gemini API for its analysis features.

-   Get your free API key from 👉 **[Google AI Studio](https://makersuite.google.com/app/apikey)**.
-   Create a new file named `.env` in the project's root directory.
-   Add your key to the `.env` file like this:

    ```env
    GEMINI_API_KEY='YOUR_API_KEY_HERE'
    ```

---

### ▶️ Running the Detector

The script is controlled via the command line.

#### Basic Examples

```bash
# Run detection on a local video file
python accident_detection.py --source demo_video.mp4

# Run detection using a live webcam
python accident_detection.py --source 0
```

#### 🎛️ Command-Line Arguments

All arguments are optional, but allow you to fine-tune the detection process.

| Argument            | Description                                                               | Default             |
| ------------------- | ------------------------------------------------------------------------- | ------------------- |
| `--source`          | Path to video file or `0` for webcam.                                     | `0`                 |
| `--model`           | Path to the YOLOv8 model weights file.                                    | `best.pt`           |
| `--output`          | Name for the generated output video.                                      | `results_video.mp4` |
| `--acc-conf`        | Confidence threshold for `vehicle_incident` classification.               | `0.75`              |
| `--frame-threshold` | Number of consecutive frames to confirm an accident.                      | `3`                 |
| `--speed-threshold` | Pixel displacement to be considered a "sudden stop".                      | `10`                |
| `--iou-threshold`   | Intersection over Union (IoU) threshold for collision detection.          | `0.1`               |
| `--location`        | Provide a location name (e.g., "Eiffel Tower") for accurate geocoding. | `None`              |

#### Advanced Example

Here's how you might run a more customized analysis:

```bash
python accident_detection.py \
    --source "path/to/your/video.mp4" \
    --model "best.pt" \
    --output "my_analysis.mp4" \
    --acc-conf 0.80 \
    --frame-threshold 4 \
    --location "1600 Amphitheatre Parkway, Mountain View, CA"
```

After running, you'll find the annotated video (`my_analysis.mp4`) and any JSON reports in the project directory.

---
import cv2
from ultralytics import YOLO
import argparse
import json
import os
import time
import torch
import google.generativeai as genai
from PIL import Image
from collections import deque, defaultdict
import numpy as np
from itertools import combinations
from utils import setup_logging, get_gps_coordinates, get_gemini_api_key

class AccidentDetector:
    def __init__(self, model_path, frame_confirmation_threshold=3, accident_report_dir='accident_reports', output_video_filename='results_video.mp4', acc_conf_thresh=0.70, other_conf_thresh=0.60, speed_threshold=10, iou_threshold=0.1, location=None):
        self.logger = setup_logging()
        self.model_path = model_path
        self.frame_confirmation_threshold = frame_confirmation_threshold
        self.accident_report_dir = accident_report_dir
        self.output_video_filename = output_video_filename
        self.acc_conf_thresh = acc_conf_thresh
        self.other_conf_thresh = other_conf_thresh
        self.speed_threshold = speed_threshold
        self.iou_threshold = iou_threshold
        self.location = location
        
        self.device = self._get_device()
        self.model = self._load_model()
        self.gemini_model = None
        self._configure_gemini()

        self.accident_frames_buffer = deque(maxlen=frame_confirmation_threshold * 5)
        self.track_history = defaultdict(lambda: deque(maxlen=10))
        
        self.consecutive_accident_details = []
        self.confirmed_accident_events = []
        self.accident_cooldown = 0

    def _configure_gemini(self):
        self.api_key = get_gemini_api_key()
        if self.api_key:
            try:
                genai.configure(api_key=self.api_key)
                self.gemini_model = genai.GenerativeModel('gemini-2.5-pro')
                self.logger.info("✅ Google Gemini 2.5 Pro AI configured successfully")
            except Exception as e:
                self.logger.error(f"Error configuring Gemini: {e}")
                self.gemini_model = None
        
    def _get_device(self):
        if torch.backends.mps.is_available():
            self.logger.info("Using Apple Metal Performance Shaders (MPS).")
            return 'mps'
        self.logger.info("MPS not available. Using CPU.")
        return 'cpu'

    def _load_model(self):
        try:
            model = YOLO(self.model_path)
            model.to(self.device)
            self.logger.info(f"Model {self.model_path} loaded on {self.device}.")
            return model
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            return None

    def generate_accident_description(self, frames):
        if not self.gemini_model:
            return {"error": "Gemini model not configured."}
        try:
            selected_frames = [frames[0], frames[len(frames)//2], frames[-1]] if len(frames) >= 3 else frames
            pil_images = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in selected_frames]
            prompt = """Analyze the traffic accident frames and return a structured JSON. Focus on the accident and participants. Format:
{
  "accident_summary": { "severity": "...", "description": "...", "inferred_sequence_of_events": ["..."] },
  "participants": [ { "type": "...", "color": "...", "visible_damage": "...", "role": "..." } ]
}
- `severity`: 'Minor', 'Moderate', 'Severe', or 'Critical'.
- `description`: One-sentence summary.
- `inferred_sequence_of_events`: Likely sequence.
- `participants`: List of involved vehicles/persons. `type`: e.g., 'Sedan', 'SUV'. `role`: e.g., 'At-fault vehicle'.
Use "Not determinable from images" if unclear.
"""
            self.logger.info(f"Generating description for {len(pil_images)} frames with Gemini 2.5 Pro...")
            response = self.gemini_model.generate_content([prompt] + pil_images, generation_config={
                "temperature": 0.2, "top_p": 0.9, "top_k": 40,
                "max_output_tokens": 4096, "response_mime_type": "application/json",
            })
            description_json = json.loads(response.text.strip().replace('```json', '').replace('```', ''))
            self.logger.info("="*60 + "\nACCIDENT DESCRIPTION (Gemini 2.5 Pro):\n" + json.dumps(description_json, indent=4) + "\n" + "="*60)
            return description_json
        except Exception as e:
            self.logger.error(f"Error with Gemini: {e}")
            return {"error": f"Description generation failed: {e}"}

    def process_video(self, video_source):
        if not self.model: return
        cap, out = None, None
        try:
            cap = cv2.VideoCapture(video_source)
            if not cap.isOpened():
                self.logger.error(f"Cannot open video source: {video_source}"); return

            fw, fh, fps = int(cap.get(3)), int(cap.get(4)), int(cap.get(5))
            out = cv2.VideoWriter(self.output_video_filename, cv2.VideoWriter_fourcc(*'mp4v'), fps, (fw, fh))

            self.logger.info(f"Processing video with tracking. Configs: acc_conf={self.acc_conf_thresh}, speed_thresh={self.speed_threshold}, iou_thresh={self.iou_threshold}. Press 'q' to quit.")

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break

                if self.accident_cooldown > 0: self.accident_cooldown -= 1
                self.accident_frames_buffer.append(frame.copy())
                results = self.model.track(frame, persist=True)
                
                accident_in_frame, method, value = self._process_frame_results(results)
                
                if accident_in_frame:
                    self.consecutive_accident_details.append({'method': method, 'value': value})
                else:
                    self.consecutive_accident_details = []

                if len(self.consecutive_accident_details) >= self.frame_confirmation_threshold and self.accident_cooldown == 0:
                    self._log_confirmed_event()
                    self.accident_cooldown = fps * 5  # 5-second cooldown
                    self.consecutive_accident_details = []

                annotated_frame = self._annotate_frame(results[0].plot())
                out.write(annotated_frame)
                cv2.imshow("Accident Detection", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'): break
        except Exception as e:
            self.logger.error(f"Unexpected error during video processing: {e}", exc_info=True)
        finally:
            if cap: cap.release()
            if out: out.release()
            cv2.destroyAllWindows()
            self._select_and_report_best_event()
            self.logger.info("Video processing completed.")

    def _log_confirmed_event(self):
        self.logger.info(f"Confirmed accident event detected after {len(self.consecutive_accident_details)} frames.")
        
        best_method, best_value = 'sudden_stop', 0.0
        priority = {'collision': 2, 'classification': 1, 'sudden_stop': 0}

        for detail in self.consecutive_accident_details:
            if priority[detail['method']] > priority[best_method]:
                best_method = detail['method']
                best_value = detail['value']
            elif priority[detail['method']] == priority[best_method] and detail['value'] > best_value:
                best_value = detail['value']
        
        self.confirmed_accident_events.append({
            'frames': list(self.accident_frames_buffer),
            'method': best_method,
            'value': best_value
        })

    def _select_and_report_best_event(self):
        if not self.confirmed_accident_events:
            self.logger.info("No accidents were detected to report.")
            return

        self.logger.info(f"Selecting the best event from {len(self.confirmed_accident_events)} confirmed accident(s).")
        
        priority = {'collision': 2, 'classification': 1, 'sudden_stop': 0}
        
        self.confirmed_accident_events.sort(key=lambda x: (priority[x['method']], x['value']), reverse=True)
        
        best_event = self.confirmed_accident_events[0]
        self.logger.info(f"Best event selected - Method: {best_event['method']}, Value: {best_event['value']:.2f}")
        
        self._handle_confirmed_accident(best_event['frames'])

    def _process_frame_results(self, results):
        if results[0].boxes is None or results[0].boxes.id is None:
            return False, 'none', 0.0

        boxes, track_ids, confs, clss = results[0].boxes.xyxy.cpu(), results[0].boxes.id.int().cpu().tolist(), results[0].boxes.conf, results[0].boxes.cls
        for box, track_id in zip(boxes, track_ids):
            self.track_history[track_id].append((int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)))

        collided_ids, max_iou = self._check_for_collisions(boxes, track_ids)
        if collided_ids:
            return True, 'collision', max_iou

        max_conf = 0.0
        incident_classified = False
        for conf, cls_id in zip(confs, clss):
            if self.model.names[int(cls_id)] == 'vehicle_incident' and conf > self.acc_conf_thresh:
                incident_classified = True
                if conf > max_conf: max_conf = float(conf)
        if incident_classified:
            return True, 'classification', max_conf

        for track_id in track_ids:
            if self._check_for_sudden_stop(track_id):
                return True, 'sudden_stop', 0.0
        
        return False, 'none', 0.0

    def _calculate_iou(self, box1, box2):
        x1_inter, y1_inter = max(box1[0], box2[0]), max(box1[1], box2[1])
        x2_inter, y2_inter = min(box1[2], box2[2]), min(box1[3], box2[3])
        inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
        if inter_area == 0: return 0.0
        box1_area, box2_area = (box1[2] - box1[0]) * (box1[3] - box1[1]), (box2[2] - box2[0]) * (box2[3] - box2[1])
        return inter_area / (box1_area + box2_area - inter_area)

    def _check_for_collisions(self, boxes, track_ids):
        collided_ids, max_iou = set(), 0.0
        track_data = list(zip(track_ids, boxes))
        for (id1, box1), (id2, box2) in combinations(track_data, 2):
            iou = self._calculate_iou(box1, box2)
            if iou > self.iou_threshold:
                if self._check_for_sudden_stop(id1) or self._check_for_sudden_stop(id2):
                    collided_ids.update([id1, id2])
                    if iou > max_iou: max_iou = iou
        return collided_ids, max_iou

    def _check_for_sudden_stop(self, track_id):
        history = self.track_history[track_id]
        if len(history) < 5: return False
        displacement = np.linalg.norm(np.array(history[-5]) - np.array(history[-1]))
        return displacement < self.speed_threshold

    def _handle_confirmed_accident(self, frames):
        lat, lon = get_gps_coordinates(location=self.location)
        description_json = self.generate_accident_description(frames)
        self._save_report(lat, lon, description_json)

    def _annotate_frame(self, frame):
        for _, centers in self.track_history.items():
            if len(centers) > 1:
                for i in range(1, len(centers)):
                    cv2.line(frame, centers[i - 1], centers[i], (0, 255, 255), 2)
        if len(self.consecutive_accident_details) > 0:
            text = f"POTENTIAL ACCIDENT... {len(self.consecutive_accident_details)}/{self.frame_confirmation_threshold}"
            cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
        if self.accident_cooldown > 0:
            cv2.putText(frame, "ACCIDENT CONFIRMED", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return frame

    def _save_report(self, lat, lon, description_json):
        if not os.path.exists(self.accident_report_dir):
            os.makedirs(self.accident_report_dir)
        
        report = {
            "timestamp": time.time(), "timestamp_readable": time.strftime("%Y-%m-%d %H:%M:%S"),
            "gps_coordinates": { "latitude": lat, "longitude": lon, "google_maps_link": f"https://www.google.com/maps?q={lat},{lon}" },
            "ai_analysis": description_json,
            "detection_parameters": {
                "confidence_threshold_accident": self.acc_conf_thresh, "frame_confirmation_threshold": self.frame_confirmation_threshold,
                "speed_threshold_for_stop": self.speed_threshold, "iou_threshold_for_collision": self.iou_threshold
            }
        }
        report_filename = os.path.join(self.accident_report_dir, f"accident_report_{int(time.time())}.json")
        with open(report_filename, 'w') as f: json.dump(report, f, indent=4)
        self.logger.info(f"\n✅ Best accident report saved to: {report_filename}")

def main():
    parser = argparse.ArgumentParser(description="Accident Detection with Google Gemini 2.5 Pro")
    parser.add_argument('--source', type=str, default='0', help="Video source ('0' for webcam).")
    parser.add_argument('--model', type=str, default='best.pt', help="Path to YOLO model.")
    parser.add_argument('--output', type=str, default='results_video.mp4', help="Output video file.")
    parser.add_argument('--acc-conf', type=float, default=0.75, help="Accident classification confidence threshold.")
    parser.add_argument('--other-conf', type=float, default=0.60, help="Other object confidence threshold.")
    parser.add_argument('--frame-threshold', type=int, default=3, help="Consecutive frames to confirm an accident.")
    parser.add_argument('--speed-threshold', type=int, default=10, help="Displacement in pixels over 5 frames to be a 'stop'.")
    parser.add_argument('--iou-threshold', type=float, default=0.1, help="IoU threshold for collision detection.")
    parser.add_argument('--location', type=str, default=None, help="Location name or address for accurate geocoding (e.g., 'Eiffel Tower').")
    args = parser.parse_args()

    setup_logging().info("Starting Accident Detection application.")
    detector = AccidentDetector(
        model_path=args.model, output_video_filename=args.output, acc_conf_thresh=args.acc_conf,
        other_conf_thresh=args.other_conf, frame_confirmation_threshold=args.frame_threshold,
        speed_threshold=args.speed_threshold, iou_threshold=args.iou_threshold, location=args.location
    )
    detector.process_video(int(args.source) if args.source.isdigit() else args.source)

if __name__ == "__main__":
    main()
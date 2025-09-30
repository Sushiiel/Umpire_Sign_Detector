

# #!/usr/bin/env python
# # -*- coding: utf-8 -*-
# """
# Hand Gesture Recognition System - Performance Optimized
# """

# import os
# import csv
# import copy
# import itertools
# from collections import Counter, deque
# import cv2 as cv
# import numpy as np
# import mediapipe as mp
# import gradio as gr
# from model import KeyPointClassifier, PointHistoryClassifier


# class HandGestureRecognizer:
#     def __init__(self):
#         print("Initializing Hand Gesture Recognizer...")
        
#         mp_hands = mp.solutions.hands
#         self.hands = mp_hands.Hands(
#     static_image_mode=False,
#     max_num_hands=1,
#     min_detection_confidence=0.3,  # Even lower
#     min_tracking_confidence=0.2,   # Even lower
#     model_complexity=0
# )
#         self.process_every_n_frames = 3
#         try:
#             self.keypoint_classifier = KeyPointClassifier()
#             self.point_history_classifier = PointHistoryClassifier()
#             print("Models loaded successfully")
#         except Exception as e:
#             print(f"Error loading models: {e}")
#             raise
        
#         self.keypoint_classifier_labels = self._load_labels(
#             'model/keypoint_classifier/keypoint_classifier_label.csv'
#         )
#         self.point_history_classifier_labels = self._load_labels(
#             'model/point_history_classifier/point_history_classifier_label.csv'
#         )
        
#         # Reduced history for speed
#         self.history_length = 8  # Was 16
#         self.point_history = deque(maxlen=self.history_length)
#         self.finger_gesture_history = deque(maxlen=self.history_length)
        
#         self.captured_samples = []
#         self.max_samples = 10
#         self.frame_count = 0
        
#         # Frame skipping for performance
#         self.process_every_n_frames = 2
#         self.last_result = None
        
#         print("Initialization complete!")
    
#     def _load_labels(self, filepath):
#         try:
#             with open(filepath, encoding='utf-8-sig') as f:
#                 reader = csv.reader(f)
#                 labels = [row[0] for row in reader]
#             print(f"Loaded {len(labels)} labels from {filepath}")
#             return labels
#         except Exception as e:
#             print(f"Error loading labels from {filepath}: {e}")
#             return ["Unknown"]
    
#     def process_frame(self, image):
#         if image is None:
#             return None, "No image received"
        
#         self.frame_count += 1
        
#         # Skip every other frame - return cached result
#         if self.frame_count % self.process_every_n_frames != 0:
#             if self.last_result is not None:
#                 return self.last_result
#             return None, "Warming up..."
        
#         try:
#             # Resize input for faster processing
#             h, w = image.shape[:2]
#             scale = 0.75  # Process at 75% resolution
#             image_small = cv.resize(image, (int(w*scale), int(h*scale)))
            
#             if len(image_small.shape) == 3 and image_small.shape[2] == 3:
#                 image_bgr = cv.cvtColor(image_small, cv.COLOR_RGB2BGR)
#             else:
#                 image_bgr = image_small
            
#             image_bgr = cv.flip(image_bgr, 1)
            
#             # No deep copy until needed
#             image_rgb = cv.cvtColor(image_bgr, cv.COLOR_BGR2RGB)
#             image_rgb.flags.writeable = False
#             results = self.hands.process(image_rgb)
#             image_rgb.flags.writeable = True
            
#             status_message = "No hands detected"
            
#             if results.multi_hand_landmarks is not None:
#                 # Only process first hand for speed
#                 hand_landmarks = results.multi_hand_landmarks[0]
#                 handedness = results.multi_handedness[0]
                
#                 brect = self._calc_bounding_rect(image_bgr, hand_landmarks)
#                 landmark_list = self._calc_landmark_list(image_bgr, hand_landmarks)
                
#                 preprocessed_landmarks = self._preprocess_landmark(landmark_list)
#                 hand_sign_id = self.keypoint_classifier(preprocessed_landmarks)
                
#                 if hand_sign_id == 2:
#                     self.point_history.append(landmark_list[8])
#                 else:
#                     self.point_history.append([0, 0])
                
#                 # Skip finger gesture classification for speed (optional)
#                 finger_gesture_id = 0
#                 preprocessed_history = self._preprocess_point_history(
#                     image_bgr, self.point_history
#                 )
#                 if len(preprocessed_history) == (self.history_length * 2):
#                     finger_gesture_id = self.point_history_classifier(
#                         preprocessed_history
#                     )
                
#                 self.finger_gesture_history.append(finger_gesture_id)
#                 most_common_gesture = Counter(
#                     self.finger_gesture_history
#                 ).most_common(1)[0][0]
                
#                 # Simplified drawing
#                 image_bgr = self._draw_landmarks_fast(image_bgr, landmark_list)
#                 image_bgr = self._draw_info_text_fast(
#                     image_bgr,
#                     brect,
#                     handedness,
#                     self.keypoint_classifier_labels[hand_sign_id]
#                 )
                
#                 hand_type = handedness.classification[0].label
#                 gesture_name = self.keypoint_classifier_labels[hand_sign_id]
#                 status_message = f"{hand_type}: {gesture_name}"
                
#                 # Capture samples less frequently
#                 if (len(self.captured_samples) < self.max_samples and 
#                     self.frame_count % 60 == 0):
#                     self._capture_sample(image_bgr, gesture_name, hand_type)
#             else:
#                 self.point_history.append([0, 0])
            
#             output_image = cv.cvtColor(image_bgr, cv.COLOR_BGR2RGB)
            
#             self.last_result = (output_image, status_message)
#             return output_image, status_message
            
#         except Exception as e:
#             error_msg = f"Processing error: {str(e)}"
#             print(error_msg)
#             return None, error_msg
    
#     def _capture_sample(self, image, gesture, hand_type):
#         sample_rgb = cv.cvtColor(copy.deepcopy(image), cv.COLOR_BGR2RGB)
#         self.captured_samples.append({
#             'image': sample_rgb,
#             'gesture': gesture,
#             'hand': hand_type
#         })
    
#     def get_captured_samples(self):
#         if len(self.captured_samples) == 0:
#             return None
#         return [
#             (sample['image'], f"{sample['hand']} - {sample['gesture']}")
#             for sample in self.captured_samples
#         ]
    
#     def clear_samples(self):
#         self.captured_samples = []
#         self.frame_count = 0
#         self.last_result = None
    
#     def _calc_bounding_rect(self, image, landmarks):
#         h, w = image.shape[:2]
#         xs = [min(int(lm.x * w), w-1) for lm in landmarks.landmark]
#         ys = [min(int(lm.y * h), h-1) for lm in landmarks.landmark]
#         x_min, x_max = min(xs), max(xs)
#         y_min, y_max = min(ys), max(ys)
#         return [x_min, y_min, x_max, y_max]
    
#     def _calc_landmark_list(self, image, landmarks):
#         h, w = image.shape[:2]
#         return [[min(int(lm.x * w), w-1), min(int(lm.y * h), h-1)] 
#                 for lm in landmarks.landmark]
    
#     def _preprocess_landmark(self, landmark_list):
#         temp_list = copy.deepcopy(landmark_list)
#         base_x, base_y = temp_list[0]
#         temp_list = [[x - base_x, y - base_y] for x, y in temp_list]
#         temp_list = list(itertools.chain.from_iterable(temp_list))
#         max_value = max(map(abs, temp_list)) or 1
#         return [v / max_value for v in temp_list]
    
#     def _preprocess_point_history(self, image, point_history):
#         h, w = image.shape[:2]
#         temp_history = copy.deepcopy(point_history)
        
#         if len(temp_history) == 0:
#             return []
        
#         base_x, base_y = temp_history[0]
#         temp_history = [
#             [(x - base_x) / w, (y - base_y) / h]
#             for x, y in temp_history
#         ]
#         return list(itertools.chain.from_iterable(temp_history))
    
#     def _draw_landmarks_fast(self, image, landmarks):
#         """Simplified drawing - just key points and palm"""
#         if len(landmarks) == 0:
#             return image
        
#         # Draw only palm outline
#         palm_indices = [0, 1, 2, 5, 9, 13, 17, 0]
#         for i in range(len(palm_indices) - 1):
#             start = landmarks[palm_indices[i]]
#             end = landmarks[palm_indices[i + 1]]
#             cv.line(image, tuple(start), tuple(end), (255, 255, 255), 1)
        
#         # Draw fingertips only
#         for i in [4, 8, 12, 16, 20]:
#             cv.circle(image, tuple(landmarks[i]), 5, (0, 255, 0), -1)
        
#         return image
    
#     def _draw_info_text_fast(self, image, brect, handedness, gesture):
#         """Simplified info display"""
#         hand_label = handedness.classification[0].label
#         info_text = f"{hand_label}: {gesture}"
        
#         # Single text overlay
#         cv.putText(image, info_text, (10, 30),
#                   cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv.LINE_AA)
        
#         return image


# # Global recognizer
# recognizer = None


# def initialize_recognizer():
#     global recognizer
#     if recognizer is None:
#         recognizer = HandGestureRecognizer()
#     return recognizer


# def process_video_frame(image):
#     global recognizer
#     if recognizer is None:
#         recognizer = initialize_recognizer()
    
#     if image is None:
#         return None, "Waiting for camera..."
    
#     return recognizer.process_frame(image)


# def show_captured_samples():
#     global recognizer
#     if recognizer is None:
#         return None
#     return recognizer.get_captured_samples()


# def clear_samples():
#     global recognizer
#     if recognizer is not None:
#         recognizer.clear_samples()
#     return None, "Samples cleared!"


# def create_interface():
#     with gr.Blocks(title="Hand Gesture Recognition") as demo:
#         gr.Markdown("# Hand Gesture Recognition")
#         gr.Markdown("Real-time gesture detection - optimized for speed")
        
#         with gr.Row():
#             with gr.Column():
#                 webcam_input = gr.Image(
#                     sources=["webcam"],
#                     streaming=True,
#                     label="Camera",
#                     type="numpy"
#                 )
#                 status_box = gr.Textbox(
#                     label="Status",
#                     value="Ready",
#                     interactive=False
#                 )
            
#             with gr.Column():
#                 output_display = gr.Image(
#                     label="Detection",
#                     type="numpy"
#                 )
        
#         with gr.Row():
#             show_samples_btn = gr.Button("Show Samples")
#             clear_samples_btn = gr.Button("Clear Samples")
        
#         sample_gallery = gr.Gallery(
#             label="Captured Samples",
#             columns=3,
#             rows=2
#         )
        
#         webcam_input.stream(
#             fn=process_video_frame,
#             inputs=webcam_input,
#             outputs=[output_display, status_box],
#             show_progress="hidden"
#         )
        
#         show_samples_btn.click(
#             fn=show_captured_samples,
#             outputs=sample_gallery
#         )
        
#         clear_samples_btn.click(
#             fn=clear_samples,
#             outputs=[sample_gallery, status_box]
#         )
    
#     return demo


# if __name__ == "__main__":
#     print("Starting optimized system...")
    
#     initialize_recognizer()
    
#     port = int(os.environ.get("PORT", 10000))
    
#     demo = create_interface()
#     demo.queue(max_size=10, default_concurrency_limit=2)
    
#     demo.launch(
#         server_name="0.0.0.0",
#         server_port=port,
#         share=False,
#         show_error=True
#     )



# """
# Cricket Umpire Signs Detector — Lean 2-Hand, Render 0.5 CPU
# - Live detection shown ONLY in the output image
# - Supports TWO hands with per-hand labels (Left / Right)
# - "Download Detected Sample" saves ONLY a detected frame (never blank)
# """

# import os
# # ---- Hard cap CPU threads for tiny Render instances ----
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# os.environ['OMP_NUM_THREADS'] = '1'
# os.environ['OPENBLAS_NUM_THREADS'] = '1'
# os.environ['MKL_NUM_THREADS'] = '1'
# os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
# os.environ['NUMEXPR_NUM_THREADS'] = '1'

# import csv
# import copy
# import itertools
# import time
# from collections import deque, defaultdict

# import cv2 as cv
# import numpy as np
# import mediapipe as mp
# import gradio as gr

# # Keep only what we use (lighter)
# from model import KeyPointClassifier  # PointHistoryClassifier not needed here

# # Try to limit OpenCV threads too
# try:
#     cv.setNumThreads(1)
# except Exception:
#     pass


# class CricketUmpireDetector:
#     def __init__(self):
#         print("\n" + "="*60)
#         print("Initializing Cricket Umpire Signs Detector (Lean 2-Hand)...")
#         print("Optimized for low CPU environment (Render 0.5 CPU)")
#         print("="*60 + "\n")

#         mp_hands = mp.solutions.hands
#         # ULTRA LOW SETTINGS + two hands
#         self.hands = mp_hands.Hands(
#             static_image_mode=False,
#             max_num_hands=2,                 # TWO hands
#             min_detection_confidence=0.3,    # permissive for speed
#             min_tracking_confidence=0.3,
#             model_complexity=0               # lightest graph
#         )

#         # Load tiny classifier
#         print("Loading AI model...")
#         self.keypoint_classifier = KeyPointClassifier()  # ensure num_threads=1 inside if TFLite
#         print("✓ Model loaded\n")

#         # Load labels
#         self.keypoint_classifier_labels = self._load_labels(
#             'model/keypoint_classifier/keypoint_classifier_label.csv'
#         )

#         # Map gestures to cricket umpire signals
#         self.umpire_signal_mapping = {
#             'Open': 'Wide',
#             'Close': 'Out',
#             'Pointer': 'No Ball',
#             'OK': 'Four',
#             'Peace': 'Six',
#             'Victory': 'Leg Bye',
#             'Rock': 'Dead Ball',
#             'Thumbs Up': 'Free Hit',
#             'Thumbs Down': 'Not Out'
#         }

#         # Runtime state
#         self.frame_count = 0
#         self.process_every_n_frames = 4     # start processing every 4th frame
#         self.last_processed_image = None

#         # Per-hand smoothing buffer (Left/Right)
#         self.smoother = defaultdict(lambda: deque(maxlen=4))

#         # Keep last detected RGB frame for download
#         self.last_detected_rgb = None
#         self.detected_in_last_frame = False

#         # Precompute fingertip indices
#         self.fingertips = (4, 8, 12, 16, 20)

#         print(f"✓ Frame skip rate: 1/{self.process_every_n_frames} (adaptive)")
#         print("="*60 + "\n")

#     def _load_labels(self, filepath):
#         try:
#             with open(filepath, encoding='utf-8-sig') as f:
#                 reader = csv.reader(f)
#                 labels = [row[0] for row in reader]
#             print(f"  Loaded {len(labels)} labels from {filepath}")
#             return labels
#         except Exception as e:
#             print(f"  Error loading labels from {filepath}: {e}")
#             return ["Unknown"]

#     def _get_umpire_signal(self, gesture_name):
#         return self.umpire_signal_mapping.get(gesture_name, gesture_name)

#     # ---------- Core processing ----------
#     def process_frame(self, image):
#         """Return only the processed/detected output image (no status text separately)."""
#         if image is None:
#             return None  # Gradio will keep last frame

#         self.frame_count += 1

#         # Skip frames for CPU efficiency — return last processed if exists
#         if self.frame_count % self.process_every_n_frames != 0:
#             return self.last_processed_image if self.last_processed_image is not None else image

#         t0 = time.time()

#         # Downscale aggressively for MediaPipe speed (320x240 target)
#         ih, iw = image.shape[:2]
#         if ih * iw > 320 * 240:
#             image_small = cv.resize(image, (320, 240), interpolation=cv.INTER_AREA)
#         else:
#             image_small = image

#         # Convert & mirror
#         bgr = cv.cvtColor(image_small, cv.COLOR_RGB2BGR)
#         bgr = cv.flip(bgr, 1)

#         # MediaPipe expects RGB
#         rgb = cv.cvtColor(bgr, cv.COLOR_BGR2RGB)
#         rgb.flags.writeable = False
#         results = self.hands.process(rgb)
#         rgb.flags.writeable = True

#         overlay = bgr  # draw on this
#         header_text_parts = []
#         detected_any = False

#         if results.multi_hand_landmarks:
#             # iterate each detected hand, aligned with handedness
#             for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
#                 hand_label = "Hand"
#                 if results.multi_handedness and i < len(results.multi_handedness):
#                     hand_label = results.multi_handedness[i].classification[0].label  # 'Left'/'Right'

#                 landmark_list = self._calc_landmark_list(overlay, hand_landmarks)
#                 pre = self._preprocess_landmark(landmark_list)
#                 hand_sign_id = self.keypoint_classifier(pre)
#                 gesture_name = self.keypoint_classifier_labels[hand_sign_id] if hand_sign_id < len(self.keypoint_classifier_labels) else "Unknown"
#                 umpire_signal = self._get_umpire_signal(gesture_name)

#                 # Smooth per hand (Left/Right streams)
#                 self.smoother[hand_label].append(umpire_signal)
#                 # Most frequent in recent buffer
#                 smoothed = max(set(self.smoother[hand_label]), key=self.smoother[hand_label].count)

#                 # Minimal drawing: fingertips + wrist dot + tiny per-hand label
#                 self._draw_minimal(overlay, landmark_list)
#                 self._put_label_near_wrist(overlay, landmark_list, f"{hand_label}: {smoothed}")

#                 header_text_parts.append(f"{hand_label}: {smoothed}")
#                 if smoothed and smoothed != "Unknown":
#                     detected_any = True
#         else:
#             header_text_parts.append("Show signals")

#         # Small header bar summarizing both hands
#         self._draw_header(overlay, " | ".join(header_text_parts))

#         # Back to RGB for Gradio
#         output_image = cv.cvtColor(overlay, cv.COLOR_BGR2RGB)
#         self.last_processed_image = output_image

#         # Store detected frame only if a real signal exists
#         self.detected_in_last_frame = detected_any
#         if detected_any:
#             self.last_detected_rgb = output_image.copy()

#         # Adaptive throttle (aim to keep under ~80–100ms per processed frame)
#         dt = time.time() - t0
#         if dt > 0.08 and self.process_every_n_frames < 6:
#             self.process_every_n_frames += 1  # skip more
#         elif dt < 0.04 and self.process_every_n_frames > 3:
#             self.process_every_n_frames -= 1  # process a bit more often

#         return output_image

#     # ---------- Drawing / helpers ----------
#     def _calc_landmark_list(self, image, landmarks):
#         h, w = image.shape[:2]
#         return [[min(int(lm.x * w), w-1), min(int(lm.y * h), h-1)]
#                 for lm in landmarks.landmark]

#     def _preprocess_landmark(self, landmark_list):
#         temp = copy.deepcopy(landmark_list)
#         base_x, base_y = temp[0]
#         temp = [[x - base_x, y - base_y] for x, y in temp]
#         temp = list(itertools.chain.from_iterable(temp))
#         max_value = max(map(abs, temp)) or 1
#         return [v / max_value for v in temp]

#     def _draw_minimal(self, image, landmarks):
#         # fingertips + wrist only (6 tiny dots)
#         for i in self.fingertips:
#             cv.circle(image, tuple(landmarks[i]), 3, (0, 255, 0), -1)
#         cv.circle(image, tuple(landmarks[0]), 4, (255, 0, 0), -1)

#     def _put_label_near_wrist(self, image, landmarks, text):
#         x, y = landmarks[0]
#         y = max(y - 10, 10)
#         cv.putText(image, text, (x, y), cv.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv.LINE_AA)

#     def _draw_header(self, image, status):
#         h, w = image.shape[:2]
#         cv.rectangle(image, (0, 0), (w, 28), (40, 40, 40), -1)
#         cv.putText(image, status, (6, 19), cv.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv.LINE_AA)

#     # ---------- Download helper ----------
#     def save_detected_sample(self):
#         """Save and return a path ONLY if we have a detected frame."""
#         if not self.detected_in_last_frame or self.last_detected_rgb is None:
#             return None
#         ts = int(time.time())
#         path = f"/tmp/umpire_detected_{ts}.png"
#         bgr = cv.cvtColor(self.last_detected_rgb, cv.COLOR_RGB2BGR)
#         cv.imwrite(path, bgr)
#         return path


# # Global detector instance
# detector = None


# def initialize_detector():
#     global detector
#     if detector is None:
#         detector = CricketUmpireDetector()
#     return detector


# def process_video_frame(image):
#     """
#     Live stream handler: returns ONLY the processed image (detected output).
#     """
#     try:
#         if detector is None:
#             initialize_detector()
#         return detector.process_frame(image)
#     except Exception as e:
#         print(f"\n❌ ERROR in process_video_frame: {e}")
#         return image  # fall back to passthrough if error


# def download_detected_sample():
#     """
#     Gradio download handler: returns a file path only when a detection exists.
#     """
#     if detector is None:
#         return None
#     return detector.save_detected_sample()


# def create_interface():
#     with gr.Blocks(title="🏏 Cricket Umpire Signs — Lean 2-Hand", theme=gr.themes.Soft()) as demo:
#         gr.Markdown("# 🏏 Cricket Umpire Signs — Lean (Two Hands)\nLive output only. Download saves a detected sample frame.")

#         with gr.Row():
#             with gr.Column(scale=1):
#                 webcam_input = gr.Image(
#                     sources=["webcam"],
#                     streaming=True,
#                     label="📹 Camera",
#                     type="numpy",
#                     mirror_webcam=False
#                 )
#             with gr.Column(scale=1):
#                 output_display = gr.Image(
#                     label="🎯 Live Detection Output",
#                     type="numpy"
#                 )

#         # Live streaming: ONLY the image output (no status textbox)
#         webcam_input.stream(
#             fn=process_video_frame,
#             inputs=webcam_input,
#             outputs=output_display,
#             show_progress="hidden"
#         )

#         # Download button — returns file ONLY when a detection exists
#         download_btn = gr.DownloadButton("📥 Download Detected Sample", file_name="umpire_detected.png")
#         download_btn.click(fn=download_detected_sample, outputs=download_btn)

#     return demo


# if __name__ == "__main__":
#     print("\n" + "="*60)
#     print("🏏 CRICKET UMPIRE SIGNS DETECTOR — Lean 2-Hand")
#     print("Optimized for Render.com (0.5 CPU)")
#     print("="*60 + "\n")

#     # Initialize detector on startup (warms up graphs)
#     initialize_detector()

#     port = int(os.environ.get("PORT", 10000))
#     print(f"\n🌐 Starting server on port {port}...")

#     demo = create_interface()
#     # Tiny queue + single concurrency to avoid CPU bursts
#     demo.queue(
#         max_size=2,
#         default_concurrency_limit=1
#     )

#     print("\n🚀 Launching Gradio interface...")
#     demo.launch(
#         server_name="0.0.0.0",
#         server_port=port,
#         share=False,
#         show_error=True
#     )

#     print("\n✓ Server running!")
#     print("="*60)

"""
Streamlit — Cricket Umpire Signs (Optimized for 0.5 CPU)
- Efficient two-hand detection with minimal resource usage
- Live webcam with gesture display
- Download button for detected frames only
"""

import os
# ---- Hard cap CPU threads for tiny instances ----
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import csv
import copy
import itertools
import time
from collections import deque, defaultdict
from typing import Optional

import av
import cv2 as cv
import numpy as np
import mediapipe as mp
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode, RTCConfiguration

# Try to keep OpenCV single-threaded
try:
    cv.setNumThreads(1)
except Exception:
    pass

# -------------------- Simple KeyPoint Classifier (Fallback) --------------------
class KeyPointClassifier:
    """Fallback classifier if your model isn't available"""
    def __init__(self):
        self.model = None
        # Simple gesture mapping based on hand landmarks
        self.gesture_thresholds = {
            'Open': lambda lms: self._is_open_hand(lms),
            'Close': lambda lms: self._is_closed_fist(lms),
            'Pointer': lambda lms: self._is_pointing(lms),
            'OK': lambda lms: self._is_ok_sign(lms),
            'Peace': lambda lms: self._is_peace_sign(lms),
            'Thumbs Up': lambda lms: self._is_thumbs_up(lms),
        }
    
    def __call__(self, features):
        # Simple rule-based classification as fallback
        # In practice, you'd load your actual trained model here
        return 0  # Default to first class
    
    def _is_open_hand(self, landmarks):
        """Check if all fingers are extended"""
        if len(landmarks) < 21:
            return False
        # Simplified check - compare fingertip y-coordinates with base y-coordinates
        try:
            tips = [8, 12, 16, 20]  # fingertip indices
            bases = [6, 10, 14, 18]  # base indices
            extended_fingers = sum(1 for tip, base in zip(tips, bases) 
                                if landmarks[tip][1] < landmarks[base][1])
            return extended_fingers >= 3
        except:
            return False
    
    def _is_closed_fist(self, landmarks):
        """Check if all fingers are closed"""
        if len(landmarks) < 21:
            return False
        try:
            tips = [8, 12, 16, 20]
            bases = [5, 9, 13, 17]
            closed_fingers = sum(1 for tip, base in zip(tips, bases) 
                               if landmarks[tip][1] > landmarks[base][1])
            return closed_fingers >= 3
        except:
            return False
    
    def _is_pointing(self, landmarks):
        """Check if only index finger is extended"""
        if len(landmarks) < 21:
            return False
        try:
            # Index finger extended, others closed
            return (landmarks[8][1] < landmarks[6][1] and  # index extended
                    landmarks[12][1] > landmarks[10][1] and  # middle closed
                    landmarks[16][1] > landmarks[14][1] and  # ring closed
                    landmarks[20][1] > landmarks[18][1])     # pinky closed
        except:
            return False
    
    def _is_ok_sign(self, landmarks):
        """Check for OK sign (thumb and index finger touching)"""
        if len(landmarks) < 21:
            return False
        try:
            thumb_tip = np.array(landmarks[4])
            index_tip = np.array(landmarks[8])
            distance = np.linalg.norm(thumb_tip - index_tip)
            return distance < 30  # Threshold for touching
        except:
            return False
    
    def _is_peace_sign(self, landmarks):
        """Check for peace sign (index and middle extended)"""
        if len(landmarks) < 21:
            return False
        try:
            return (landmarks[8][1] < landmarks[6][1] and  # index extended
                    landmarks[12][1] < landmarks[10][1] and  # middle extended
                    landmarks[16][1] > landmarks[14][1] and  # ring closed
                    landmarks[20][1] > landmarks[18][1])     # pinky closed
        except:
            return False
    
    def _is_thumbs_up(self, landmarks):
        """Check for thumbs up"""
        if len(landmarks) < 21:
            return False
        try:
            return (landmarks[4][1] < landmarks[3][1] and  # thumb extended up
                    landmarks[8][1] > landmarks[6][1] and  # other fingers closed
                    landmarks[12][1] > landmarks[10][1] and
                    landmarks[16][1] > landmarks[14][1] and
                    landmarks[20][1] > landmarks[18][1])
        except:
            return False

# -------------------- Efficient Detector --------------------
class EfficientUmpireDetector:
    def __init__(self):
        # Initialize MediaPipe Hands with minimal settings
        mp_hands = mp.solutions.hands
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.4,  # Slightly higher for stability
            min_tracking_confidence=0.3,
            model_complexity=0,  # Lowest complexity
        )
        
        # Initialize classifier
        try:
            # Try to import your actual model
            from model import KeyPointClassifier as ActualClassifier
            self.keypoint_classifier = ActualClassifier()
            st.success("✓ Loaded trained gesture classifier")
        except ImportError:
            self.keypoint_classifier = KeyPointClassifier()
            st.warning("⚠ Using fallback gesture classifier")
        
        # Load labels
        self.labels = self._load_labels()
        
        # Gesture to cricket signal mapping
        self.map_dict = {
            'Open': 'Wide',
            'Close': 'Out', 
            'Pointer': 'No Ball',
            'OK': 'Four',
            'Peace': 'Six',
            'Victory': 'Leg Bye',
            'Rock': 'Dead Ball',
            'Thumbs Up': 'Free Hit',
            'Thumbs Down': 'Not Out'
        }
        
        # Key landmarks for visualization
        self.fingertips = [4, 8, 12, 16, 20]  # thumb, index, middle, ring, pinky
        self.connections = mp.solutions.hands.HAND_CONNECTIONS

    def _load_labels(self):
        """Load gesture labels from CSV"""
        default_labels = ["Open", "Close", "Pointer", "OK", "Peace", "Rock", "Thumbs Up", "Thumbs Down"]
        try:
            with open('model/keypoint_classifier/keypoint_classifier_label.csv', encoding='utf-8-sig') as f:
                return [row[0] for row in csv.reader(f)]
        except Exception as e:
            st.warning(f"Could not load labels: {e}. Using default labels.")
            return default_labels

    def _calc_landmarks(self, img, landmarks):
        """Convert normalized landmarks to pixel coordinates"""
        h, w = img.shape[:2]
        return [[int(lm.x * w), int(lm.y * h)] for lm in landmarks.landmark]

    def _preprocess_landmarks(self, landmark_list):
        """Normalize landmarks for classification"""
        if not landmark_list:
            return []
            
        temp = copy.deepcopy(landmark_list)
        
        # Convert to relative coordinates
        base_x, base_y = temp[0]
        temp = [[x - base_x, y - base_y] for x, y in temp]
        
        # Flatten and normalize
        temp = list(itertools.chain.from_iterable(temp))
        max_val = max(map(abs, temp)) or 1
        
        return [v / max_val for v in temp]

    def _draw_hand_landmarks(self, img, landmarks, hand_type="Hand"):
        """Efficiently draw hand landmarks and connections"""
        # Draw connections
        for connection in self.connections:
            start_idx, end_idx = connection
            if start_idx < len(landmarks) and end_idx < len(landmarks):
                cv.line(img, tuple(landmarks[start_idx]), tuple(landmarks[end_idx]), 
                       (0, 255, 0), 1, cv.LINE_AA)
        
        # Draw key points
        for idx in self.fingertips:
            if idx < len(landmarks):
                cv.circle(img, tuple(landmarks[idx]), 4, (0, 255, 0), -1, cv.LINE_AA)
        
        # Draw wrist point
        if landmarks:
            cv.circle(img, tuple(landmarks[0]), 5, (255, 0, 0), -1, cv.LINE_AA)

    def _add_gesture_label(self, img, landmarks, gesture, hand_type):
        """Add gesture label above hand"""
        if not landmarks:
            return
            
        x, y = landmarks[0]  # Use wrist position as anchor
        label_y = max(y - 15, 20)
        
        label_text = f"{hand_type}: {gesture}"
        cv.putText(img, label_text, (x, label_y), 
                  cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv.LINE_AA)

    def _add_header(self, img, signals):
        """Add header with current signals"""
        h, w = img.shape[:2]
        
        # Semi-transparent header background
        overlay = img.copy()
        cv.rectangle(overlay, (0, 0), (w, 35), (40, 40, 40), -1)
        cv.addWeighted(overlay, 0.7, img, 0.3, 0, img)
        
        header_text = " | ".join(signals) if signals else "Show hand signals"
        cv.putText(img, header_text, (10, 25), 
                  cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv.LINE_AA)

    def process_frame(self, frame_rgb, smoother, frame_count):
        """
        Process a single frame for hand detection and gesture recognition
        Returns: (processed_image, detected_gestures, any_detection)
        """
        # Resize for efficiency (maintain aspect ratio)
        h, w = frame_rgb.shape[:2]
        if w > 320:
            scale = 320 / w
            new_w, new_h = 320, int(h * scale)
            frame_rgb = cv.resize(frame_rgb, (new_w, new_h), interpolation=cv.INTER_AREA)
        
        # Flip for mirror effect and convert to BGR for MediaPipe
        frame_bgr = cv.cvtColor(cv.flip(frame_rgb, 1), cv.COLOR_RGB2BGR)
        
        # Process with MediaPipe Hands
        rgb_for_mp = cv.cvtColor(frame_bgr, cv.COLOR_BGR2RGB)
        rgb_for_mp.flags.writeable = False
        results = self.hands.process(rgb_for_mp)
        rgb_for_mp.flags.writeable = True
        
        output_img = frame_bgr.copy()
        current_signals = []
        detected_any = False
        
        if results.multi_hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Get hand type (Left/Right)
                hand_type = "Right"  # Default
                if results.multi_handedness and hand_idx < len(results.multi_handedness):
                    hand_type = results.multi_handedness[hand_idx].classification[0].label
                
                # Convert landmarks to pixel coordinates
                landmarks = self._calc_landmarks(output_img, hand_landmarks)
                
                # Classify gesture
                preprocessed = self._preprocess_landmarks(landmarks)
                if preprocessed:
                    try:
                        gesture_id = self.keypoint_classifier(preprocessed)
                        gesture_name = self.labels[gesture_id] if gesture_id < len(self.labels) else "Unknown"
                    except:
                        gesture_name = "Unknown"
                    
                    # Map to cricket signal
                    cricket_signal = self.map_dict.get(gesture_name, gesture_name)
                    
                    # Apply smoothing
                    smoother[hand_type].append(cricket_signal)
                    if len(smoother[hand_type]) > 5:  # Keep last 5 detections
                        smoother[hand_type].popleft()
                    
                    # Get most frequent recent signal
                    if smoother[hand_type]:
                        smoothed_signal = max(set(smoother[hand_type]), 
                                            key=lambda x: list(smoother[hand_type]).count(x))
                    else:
                        smoothed_signal = cricket_signal
                    
                    # Visualize
                    self._draw_hand_landmarks(output_img, landmarks, hand_type)
                    self._add_gesture_label(output_img, landmarks, smoothed_signal, hand_type)
                    
                    current_signals.append(f"{hand_type}: {smoothed_signal}")
                    
                    if smoothed_signal != "Unknown":
                        detected_any = True
                        st.sidebar.success(f"🎯 Detected: {hand_type} - {smoothed_signal}")
        
        # Add header with current signals
        self._add_header(output_img, current_signals)
        
        # Convert back to RGB for Streamlit
        output_rgb = cv.cvtColor(output_img, cv.COLOR_BGR2RGB)
        
        return output_rgb, current_signals, detected_any


# -------------------- Video Processor --------------------
class UmpireVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.detector = EfficientUmpireDetector()
        self.smoother = defaultdict(lambda: deque(maxlen=5))  # Per-hand smoothing
        self.frame_count = 0
        self.process_every = 3  # Process every 3rd frame for efficiency
        self.last_output = None
        self.last_detected_frame = None
        self.last_detection_time = 0
        
        # Statistics
        self.processed_frames = 0
        self.detection_count = 0

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        self.frame_count += 1
        rgb_frame = frame.to_ndarray(format="rgb24")
        
        # Skip frames for CPU efficiency
        if self.frame_count % self.process_every != 0 and self.last_output is not None:
            return av.VideoFrame.from_ndarray(self.last_output, format="rgb24")
        
        # Process frame
        start_time = time.time()
        processed_frame, signals, detected = self.detector.process_frame(
            rgb_frame, self.smoother, self.frame_count
        )
        
        self.processed_frames += 1
        if detected:
            self.detection_count += 1
            self.last_detected_frame = processed_frame.copy()
            self.last_detection_time = time.time()
        
        self.last_output = processed_frame
        
        # Adaptive frame skipping based on processing time
        processing_time = time.time() - start_time
        if processing_time > 0.1:  # If processing takes too long
            self.process_every = min(6, self.process_every + 1)
        elif processing_time < 0.05 and self.process_every > 2:  # If we have headroom
            self.process_every = max(2, self.process_every - 1)
        
        return av.VideoFrame.from_ndarray(processed_frame, format="rgb24")


# -------------------- Streamlit UI --------------------
st.set_page_config(
    page_title="🏏 Cricket Umpire Signs Detector",
    page_icon="🏏",
    layout="wide"
)

st.title("🏏 Cricket Umpire Signs Detector")
st.markdown("""
Real-time hand gesture detection for cricket umpire signals using your webcam. 
Optimized for low CPU usage (0.5 cores).
""")

# Sidebar with instructions and stats
with st.sidebar:
    st.header("ℹ️ Instructions")
    st.markdown("""
    1. Allow camera access when prompted
    2. Show clear hand gestures to the camera
    3. Supported gestures:
       - ✋ Open Hand → **Wide**
       - ✊ Closed Fist → **Out**
       - 👆 Pointing → **No Ball**
       - 👌 OK Sign → **Four**
       - ✌️ Peace Sign → **Six**
       - 👍 Thumbs Up → **Free Hit**
    """)
    
    st.header("📊 Detection Info")
    if 'processor' in st.session_state:
        proc = st.session_state.processor
        st.metric("Processed Frames", proc.processed_frames)
        st.metric("Successful Detections", proc.detection_count)
        if proc.last_detection_time > 0:
            st.metric("Last Detection", f"{time.time() - proc.last_detection_time:.1f}s ago")

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📹 Live Camera Feed")
    st.markdown("Position your hands clearly in the frame")

with col2:
    st.subheader("🎯 Detection Output")
    st.markdown("Gesture recognition results will appear here")

st.divider()

# WebRTC streamer configuration
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]}
    ]
})

# Initialize or get processor instance
def get_video_processor():
    if 'processor' not in st.session_state:
        st.session_state.processor = UmpireVideoProcessor()
    return st.session_state.processor

# WebRTC streamer
webrtc_ctx = webrtc_streamer(
    key="umpire-signs-detector",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=get_video_processor,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={
        "video": {
            "width": {"ideal": 640},
            "height": {"ideal": 480},
            "frameRate": {"ideal": 15, "max": 20}  # Lower FPS for efficiency
        },
        "audio": False
    },
    async_processing=True,
)

# Download section
st.subheader("💾 Save Detected Signals")
st.markdown("Download frames where umpire signals were successfully detected.")

if webrtc_ctx.state.playing:
    processor = get_video_processor()
    
    if processor.last_detected_frame is not None:
        # Convert the detected frame for download
        download_frame = cv.cvtColor(processor.last_detected_frame, cv.COLOR_RGB2BGR)
        success, buffer = cv.imencode(".png", download_frame)
        
        if success:
            st.download_button(
                label="📥 Download Last Detected Frame",
                data=buffer.tobytes(),
                file_name=f"umpire_signal_{int(time.time())}.png",
                mime="image/png",
                help="Saves the most recent frame with a detected umpire signal"
            )
            st.success("✅ Frame with detected signal is ready for download!")
        else:
            st.error("❌ Could not encode image for download")
    else:
        st.info("👆 Perform a clear hand signal to enable download")
else:
    st.info("🎥 Click 'START' to begin camera feed and gesture detection")

# Footer
st.divider()
st.caption(
    "Built with MediaPipe, OpenCV, and Streamlit • "
    "Optimized for low-resource deployment • "
    "Gesture detection may vary based on lighting and camera quality"
)



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
Streamlit — Cricket Umpire Signs (Lean 2-Hand, 0.5 CPU)
- Live webcam with minimal overlay
- Two-hand detection (Left/Right)
- Download button saves ONLY a detected frame
"""

import os
# ---- Hard cap CPU threads for tiny Render instances ----
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
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode

# import your classifier (ensure it uses 1 thread if TFLite)
from model import KeyPointClassifier  # keep your existing module/folder


# Try to keep OpenCV single-threaded
try:
    cv.setNumThreads(1)
except Exception:
    pass


# -------------------- Detector (stateless helper) --------------------
class LeanUmpireDetector:
    def __init__(self):
        mp_hands = mp.solutions.hands
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3,
            model_complexity=0,
        )
        self.keypoint_classifier = KeyPointClassifier()
        self.labels = self._load_labels('model/keypoint_classifier/keypoint_classifier_label.csv')
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
        self.fingertips = (4, 8, 12, 16, 20)

    def _load_labels(self, path):
        try:
            with open(path, encoding='utf-8-sig') as f:
                return [row[0] for row in csv.reader(f)]
        except Exception:
            return ["Unknown"]

    def _map(self, g):
        return self.map_dict.get(g, g)

    def _calc_landmarks(self, img, landmarks):
        h, w = img.shape[:2]
        return [[min(int(lm.x * w), w-1), min(int(lm.y * h), h-1)]
                for lm in landmarks.landmark]

    def _preprocess(self, lm_list):
        temp = copy.deepcopy(lm_list)
        bx, by = temp[0]
        temp = [[x - bx, y - by] for x, y in temp]
        temp = list(itertools.chain.from_iterable(temp))
        mv = max(map(abs, temp)) or 1
        return [v / mv for v in temp]

    def _draw_points(self, img, lms):
        for i in self.fingertips:
            cv.circle(img, tuple(lms[i]), 3, (0, 255, 0), -1)
        cv.circle(img, tuple(lms[0]), 4, (255, 0, 0), -1)

    def _label(self, img, lms, txt):
        x, y = lms[0]
        y = max(y - 10, 10)
        cv.putText(img, txt, (x, y), cv.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv.LINE_AA)

    def _header(self, img, text):
        h, w = img.shape[:2]
        cv.rectangle(img, (0, 0), (w, 28), (40, 40, 40), -1)
        cv.putText(img, text, (6, 19), cv.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv.LINE_AA)

    def run(self, frame_rgb: np.ndarray, smoother, process_every: int, frame_i: int):
        """
        Returns (output_rgb, detected_any)
        - frame skipping, 320x240 resize, minimal draw
        """
        # Skip logic handled by transformer; here we just process
        img = frame_rgb
        ih, iw = img.shape[:2]
        if ih * iw > 320 * 240:
            img = cv.resize(img, (320, 240), interpolation=cv.INTER_AREA)

        bgr = cv.cvtColor(img, cv.COLOR_RGB2BGR)
        bgr = cv.flip(bgr, 1)

        rgb = cv.cvtColor(bgr, cv.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        res = self.hands.process(rgb)
        rgb.flags.writeable = True

        overlay = bgr
        header = []
        detected = False

        if res.multi_hand_landmarks:
            for i, hand_lms in enumerate(res.multi_hand_landmarks):
                hand_label = "Hand"
                if res.multi_handedness and i < len(res.multi_handedness):
                    hand_label = res.multi_handedness[i].classification[0].label  # Left/Right

                lms = self._calc_landmarks(overlay, hand_lms)
                feats = self._preprocess(lms)
                hid = self.keypoint_classifier(feats)
                gname = self.labels[hid] if hid < len(self.labels) else "Unknown"
                signal = self._map(gname)

                # Smooth per-hand
                smoother[hand_label].append(signal)
                smoothed = max(set(smoother[hand_label]), key=smoother[hand_label].count)

                self._draw_points(overlay, lms)
                self._label(overlay, lms, f"{hand_label}: {smoothed}")

                header.append(f"{hand_label}: {smoothed}")
                if smoothed and smoothed != "Unknown":
                    detected = True
        else:
            header.append("Show signals")

        self._header(overlay, " | ".join(header))
        out_rgb = cv.cvtColor(overlay, cv.COLOR_BGR2RGB)
        return out_rgb, detected


# -------------------- Video Transformer --------------------
class Transformer(VideoTransformerBase):
    def __init__(self):
        self.detector = LeanUmpireDetector()
        self.smoother = defaultdict(lambda: deque(maxlen=4))
        self.frame_i = 0
        self.process_every = 4           # adaptive 4..6
        self.last_output: Optional[np.ndarray] = None
        self.last_detected: Optional[np.ndarray] = None

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        self.frame_i += 1
        img = frame.to_ndarray(format="rgb24")

        # Frame skipping for CPU: return cached when skipping
        if self.frame_i % self.process_every != 0 and self.last_output is not None:
            return av.VideoFrame.from_ndarray(self.last_output, format="rgb24")

        t0 = time.time()
        out_rgb, detected = self.detector.run(img, self.smoother, self.process_every, self.frame_i)
        self.last_output = out_rgb
        if detected:
            self.last_detected = out_rgb.copy()

        # Adaptive throttle
        dt = time.time() - t0
        if dt > 0.08 and self.process_every < 6:
            self.process_every += 1
        elif dt < 0.04 and self.process_every > 3:
            self.process_every -= 1

        return av.VideoFrame.from_ndarray(out_rgb, format="rgb24")


# -------------------- Streamlit UI --------------------
st.set_page_config(page_title="🏏 Umpire Signs — Streamlit (Lean 2-Hand)", layout="wide")

st.title("🏏 Cricket Umpire Signs — Streamlit (Two Hands, 0.5 CPU)")
st.caption("Live detected output only. Download returns a detected sample frame (never blank).")

# Create/hold a transformer in session state so we can access last_detected
if "transformer" not in st.session_state:
    st.session_state.transformer = Transformer()

col1, col2 = st.columns([1, 1])
with col1:
    st.markdown("### 📹 Camera")
with col2:
    st.markdown("### 🎯 Live Detection Output")

webrtc_ctx = webrtc_streamer(
    key="umpire-two-hand",
    mode=WebRtcMode.SENDRECV,
    video_transformer_factory=lambda: st.session_state.transformer,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

st.divider()

# Download detected sample only
detected_img = st.session_state.transformer.last_detected
if detected_img is not None:
    # encode as PNG in-memory
    success, buf = cv.imencode(".png", cv.cvtColor(detected_img, cv.COLOR_RGB2BGR))
    if success:
        st.download_button(
            "📥 Download Detected Sample",
            data=buf.tobytes(),
            file_name="umpire_detected.png",
            mime="image/png",
            help="Saves the latest frame where at least one real signal was detected."
        )
else:
    st.info("Perform a clear signal (e.g., Open/Close/OK/Peace). The button appears once a detection is made.")

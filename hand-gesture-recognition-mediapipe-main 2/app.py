

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Hand Gesture Recognition System - Performance Optimized
"""

import os
import csv
import copy
import itertools
from collections import Counter, deque
import cv2 as cv
import numpy as np
import mediapipe as mp
import gradio as gr
from model import KeyPointClassifier, PointHistoryClassifier


class HandGestureRecognizer:
    def __init__(self):
        print("Initializing Hand Gesture Recognizer...")
        
        mp_hands = mp.solutions.hands
        self.hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.3,  # Even lower
    min_tracking_confidence=0.2,   # Even lower
    model_complexity=0
)
        self.process_every_n_frames = 3
        try:
            self.keypoint_classifier = KeyPointClassifier()
            self.point_history_classifier = PointHistoryClassifier()
            print("Models loaded successfully")
        except Exception as e:
            print(f"Error loading models: {e}")
            raise
        
        self.keypoint_classifier_labels = self._load_labels(
            'model/keypoint_classifier/keypoint_classifier_label.csv'
        )
        self.point_history_classifier_labels = self._load_labels(
            'model/point_history_classifier/point_history_classifier_label.csv'
        )
        
        # Reduced history for speed
        self.history_length = 8  # Was 16
        self.point_history = deque(maxlen=self.history_length)
        self.finger_gesture_history = deque(maxlen=self.history_length)
        
        self.captured_samples = []
        self.max_samples = 10
        self.frame_count = 0
        
        # Frame skipping for performance
        self.process_every_n_frames = 2
        self.last_result = None
        
        print("Initialization complete!")
    
    def _load_labels(self, filepath):
        try:
            with open(filepath, encoding='utf-8-sig') as f:
                reader = csv.reader(f)
                labels = [row[0] for row in reader]
            print(f"Loaded {len(labels)} labels from {filepath}")
            return labels
        except Exception as e:
            print(f"Error loading labels from {filepath}: {e}")
            return ["Unknown"]
    
    def process_frame(self, image):
        if image is None:
            return None, "No image received"
        
        self.frame_count += 1
        
        # Skip every other frame - return cached result
        if self.frame_count % self.process_every_n_frames != 0:
            if self.last_result is not None:
                return self.last_result
            return None, "Warming up..."
        
        try:
            # Resize input for faster processing
            h, w = image.shape[:2]
            scale = 0.75  # Process at 75% resolution
            image_small = cv.resize(image, (int(w*scale), int(h*scale)))
            
            if len(image_small.shape) == 3 and image_small.shape[2] == 3:
                image_bgr = cv.cvtColor(image_small, cv.COLOR_RGB2BGR)
            else:
                image_bgr = image_small
            
            image_bgr = cv.flip(image_bgr, 1)
            
            # No deep copy until needed
            image_rgb = cv.cvtColor(image_bgr, cv.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False
            results = self.hands.process(image_rgb)
            image_rgb.flags.writeable = True
            
            status_message = "No hands detected"
            
            if results.multi_hand_landmarks is not None:
                # Only process first hand for speed
                hand_landmarks = results.multi_hand_landmarks[0]
                handedness = results.multi_handedness[0]
                
                brect = self._calc_bounding_rect(image_bgr, hand_landmarks)
                landmark_list = self._calc_landmark_list(image_bgr, hand_landmarks)
                
                preprocessed_landmarks = self._preprocess_landmark(landmark_list)
                hand_sign_id = self.keypoint_classifier(preprocessed_landmarks)
                
                if hand_sign_id == 2:
                    self.point_history.append(landmark_list[8])
                else:
                    self.point_history.append([0, 0])
                
                # Skip finger gesture classification for speed (optional)
                finger_gesture_id = 0
                preprocessed_history = self._preprocess_point_history(
                    image_bgr, self.point_history
                )
                if len(preprocessed_history) == (self.history_length * 2):
                    finger_gesture_id = self.point_history_classifier(
                        preprocessed_history
                    )
                
                self.finger_gesture_history.append(finger_gesture_id)
                most_common_gesture = Counter(
                    self.finger_gesture_history
                ).most_common(1)[0][0]
                
                # Simplified drawing
                image_bgr = self._draw_landmarks_fast(image_bgr, landmark_list)
                image_bgr = self._draw_info_text_fast(
                    image_bgr,
                    brect,
                    handedness,
                    self.keypoint_classifier_labels[hand_sign_id]
                )
                
                hand_type = handedness.classification[0].label
                gesture_name = self.keypoint_classifier_labels[hand_sign_id]
                status_message = f"{hand_type}: {gesture_name}"
                
                # Capture samples less frequently
                if (len(self.captured_samples) < self.max_samples and 
                    self.frame_count % 60 == 0):
                    self._capture_sample(image_bgr, gesture_name, hand_type)
            else:
                self.point_history.append([0, 0])
            
            output_image = cv.cvtColor(image_bgr, cv.COLOR_BGR2RGB)
            
            self.last_result = (output_image, status_message)
            return output_image, status_message
            
        except Exception as e:
            error_msg = f"Processing error: {str(e)}"
            print(error_msg)
            return None, error_msg
    
    def _capture_sample(self, image, gesture, hand_type):
        sample_rgb = cv.cvtColor(copy.deepcopy(image), cv.COLOR_BGR2RGB)
        self.captured_samples.append({
            'image': sample_rgb,
            'gesture': gesture,
            'hand': hand_type
        })
    
    def get_captured_samples(self):
        if len(self.captured_samples) == 0:
            return None
        return [
            (sample['image'], f"{sample['hand']} - {sample['gesture']}")
            for sample in self.captured_samples
        ]
    
    def clear_samples(self):
        self.captured_samples = []
        self.frame_count = 0
        self.last_result = None
    
    def _calc_bounding_rect(self, image, landmarks):
        h, w = image.shape[:2]
        xs = [min(int(lm.x * w), w-1) for lm in landmarks.landmark]
        ys = [min(int(lm.y * h), h-1) for lm in landmarks.landmark]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        return [x_min, y_min, x_max, y_max]
    
    def _calc_landmark_list(self, image, landmarks):
        h, w = image.shape[:2]
        return [[min(int(lm.x * w), w-1), min(int(lm.y * h), h-1)] 
                for lm in landmarks.landmark]
    
    def _preprocess_landmark(self, landmark_list):
        temp_list = copy.deepcopy(landmark_list)
        base_x, base_y = temp_list[0]
        temp_list = [[x - base_x, y - base_y] for x, y in temp_list]
        temp_list = list(itertools.chain.from_iterable(temp_list))
        max_value = max(map(abs, temp_list)) or 1
        return [v / max_value for v in temp_list]
    
    def _preprocess_point_history(self, image, point_history):
        h, w = image.shape[:2]
        temp_history = copy.deepcopy(point_history)
        
        if len(temp_history) == 0:
            return []
        
        base_x, base_y = temp_history[0]
        temp_history = [
            [(x - base_x) / w, (y - base_y) / h]
            for x, y in temp_history
        ]
        return list(itertools.chain.from_iterable(temp_history))
    
    def _draw_landmarks_fast(self, image, landmarks):
        """Simplified drawing - just key points and palm"""
        if len(landmarks) == 0:
            return image
        
        # Draw only palm outline
        palm_indices = [0, 1, 2, 5, 9, 13, 17, 0]
        for i in range(len(palm_indices) - 1):
            start = landmarks[palm_indices[i]]
            end = landmarks[palm_indices[i + 1]]
            cv.line(image, tuple(start), tuple(end), (255, 255, 255), 1)
        
        # Draw fingertips only
        for i in [4, 8, 12, 16, 20]:
            cv.circle(image, tuple(landmarks[i]), 5, (0, 255, 0), -1)
        
        return image
    
    def _draw_info_text_fast(self, image, brect, handedness, gesture):
        """Simplified info display"""
        hand_label = handedness.classification[0].label
        info_text = f"{hand_label}: {gesture}"
        
        # Single text overlay
        cv.putText(image, info_text, (10, 30),
                  cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv.LINE_AA)
        
        return image


# Global recognizer
recognizer = None


def initialize_recognizer():
    global recognizer
    if recognizer is None:
        recognizer = HandGestureRecognizer()
    return recognizer


def process_video_frame(image):
    global recognizer
    if recognizer is None:
        recognizer = initialize_recognizer()
    
    if image is None:
        return None, "Waiting for camera..."
    
    return recognizer.process_frame(image)


def show_captured_samples():
    global recognizer
    if recognizer is None:
        return None
    return recognizer.get_captured_samples()


def clear_samples():
    global recognizer
    if recognizer is not None:
        recognizer.clear_samples()
    return None, "Samples cleared!"


def create_interface():
    with gr.Blocks(title="Hand Gesture Recognition") as demo:
        gr.Markdown("# Hand Gesture Recognition")
        gr.Markdown("Real-time gesture detection - optimized for speed")
        
        with gr.Row():
            with gr.Column():
                webcam_input = gr.Image(
                    sources=["webcam"],
                    streaming=True,
                    label="Camera",
                    type="numpy"
                )
                status_box = gr.Textbox(
                    label="Status",
                    value="Ready",
                    interactive=False
                )
            
            with gr.Column():
                output_display = gr.Image(
                    label="Detection",
                    type="numpy"
                )
        
        with gr.Row():
            show_samples_btn = gr.Button("Show Samples")
            clear_samples_btn = gr.Button("Clear Samples")
        
        sample_gallery = gr.Gallery(
            label="Captured Samples",
            columns=3,
            rows=2
        )
        
        webcam_input.stream(
            fn=process_video_frame,
            inputs=webcam_input,
            outputs=[output_display, status_box],
            show_progress="hidden"
        )
        
        show_samples_btn.click(
            fn=show_captured_samples,
            outputs=sample_gallery
        )
        
        clear_samples_btn.click(
            fn=clear_samples,
            outputs=[sample_gallery, status_box]
        )
    
    return demo


if __name__ == "__main__":
    print("Starting optimized system...")
    
    initialize_recognizer()
    
    port = int(os.environ.get("PORT", 10000))
    
    demo = create_interface()
    demo.queue(max_size=10, default_concurrency_limit=2)
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False,
        show_error=True
    )



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



#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Cricket Umpire Signs Detector - Optimized for Real-time Detection
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


class CricketUmpireDetector:
    def __init__(self):
        print("Initializing Cricket Umpire Signs Detector...")
        
        mp_hands = mp.solutions.hands
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,  # Both hands for umpire signals
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1
        )
        
        try:
            self.keypoint_classifier = KeyPointClassifier()
            self.point_history_classifier = PointHistoryClassifier()
            print("Models loaded successfully")
        except Exception as e:
            print(f"Error loading models: {e}")
            raise
        
        # Load labels
        self.keypoint_classifier_labels = self._load_labels(
            'model/keypoint_classifier/keypoint_classifier_label.csv'
        )
        self.point_history_classifier_labels = self._load_labels(
            'model/point_history_classifier/point_history_classifier_label.csv'
        )
        
        # Map gestures to cricket umpire signals
        self.umpire_signal_mapping = {
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
        
        # History for tracking
        self.history_length = 16
        self.point_history = deque(maxlen=self.history_length)
        self.finger_gesture_history = deque(maxlen=self.history_length)
        
        # Captured samples storage
        self.captured_samples = []
        self.max_samples = 15
        self.frame_count = 0
        self.last_detection_frame = 0
        self.min_frames_between_capture = 30  # Capture every 30 frames minimum
        
        # Detection state
        self.last_result = None
        self.current_detection = None
        self.detection_confidence_threshold = 3  # Need 3 consistent frames
        self.consistent_detection_count = 0
        
        print("Cricket Umpire Detector initialized successfully!")
    
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
    
    def _get_umpire_signal(self, gesture_name):
        """Map gesture to cricket umpire signal"""
        return self.umpire_signal_mapping.get(gesture_name, gesture_name)
    
    def process_frame(self, image):
        if image is None:
            return None, "No camera input"
        
        self.frame_count += 1
        
        try:
            # Convert image properly
            if len(image.shape) == 3 and image.shape[2] == 3:
                image_bgr = cv.cvtColor(image, cv.COLOR_RGB2BGR)
            else:
                image_bgr = image
            
            # Flip for mirror effect
            image_bgr = cv.flip(image_bgr, 1)
            debug_image = copy.deepcopy(image_bgr)
            
            # Process with MediaPipe
            image_rgb = cv.cvtColor(image_bgr, cv.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False
            results = self.hands.process(image_rgb)
            image_rgb.flags.writeable = True
            
            status_message = "Waiting for umpire signal..."
            detected_signal = None
            
            if results.multi_hand_landmarks is not None:
                for hand_landmarks, handedness in zip(
                    results.multi_hand_landmarks,
                    results.multi_handedness
                ):
                    # Calculate bounding box and landmarks
                    brect = self._calc_bounding_rect(debug_image, hand_landmarks)
                    landmark_list = self._calc_landmark_list(debug_image, hand_landmarks)
                    
                    # Preprocess and classify
                    preprocessed_landmarks = self._preprocess_landmark(landmark_list)
                    hand_sign_id = self.keypoint_classifier(preprocessed_landmarks)
                    
                    # Track point history for dynamic gestures
                    if hand_sign_id == 2:  # Pointer gesture
                        self.point_history.append(landmark_list[8])
                    else:
                        self.point_history.append([0, 0])
                    
                    # Finger gesture classification
                    finger_gesture_id = 0
                    preprocessed_history = self._preprocess_point_history(
                        debug_image, self.point_history
                    )
                    if len(preprocessed_history) == (self.history_length * 2):
                        finger_gesture_id = self.point_history_classifier(
                            preprocessed_history
                        )
                    
                    self.finger_gesture_history.append(finger_gesture_id)
                    
                    # Draw landmarks
                    debug_image = self._draw_landmarks(debug_image, landmark_list)
                    debug_image = self._draw_bounding_rect(debug_image, brect)
                    
                    # Get gesture name and umpire signal
                    gesture_name = self.keypoint_classifier_labels[hand_sign_id]
                    umpire_signal = self._get_umpire_signal(gesture_name)
                    hand_type = handedness.classification[0].label
                    
                    # Draw info
                    debug_image = self._draw_info_text(
                        debug_image,
                        brect,
                        handedness,
                        umpire_signal,
                        gesture_name
                    )
                    
                    detected_signal = umpire_signal
                    status_message = f"✓ Detected: {umpire_signal} ({hand_type} hand)"
                    
                    # Track consistent detection
                    if detected_signal == self.current_detection:
                        self.consistent_detection_count += 1
                    else:
                        self.current_detection = detected_signal
                        self.consistent_detection_count = 1
                    
                    # Capture sample if detection is consistent and timing is right
                    frames_since_last = self.frame_count - self.last_detection_frame
                    if (self.consistent_detection_count >= self.detection_confidence_threshold and
                        frames_since_last >= self.min_frames_between_capture and
                        len(self.captured_samples) < self.max_samples and
                        detected_signal != "Unknown"):
                        
                        self._capture_sample(
                            debug_image,
                            umpire_signal,
                            gesture_name,
                            hand_type
                        )
                        self.last_detection_frame = self.frame_count
                        status_message += " [SAVED]"
            else:
                self.point_history.append([0, 0])
                self.current_detection = None
                self.consistent_detection_count = 0
            
            # Draw header
            debug_image = self._draw_header(debug_image, status_message)
            
            # Convert back to RGB for display
            output_image = cv.cvtColor(debug_image, cv.COLOR_BGR2RGB)
            
            self.last_result = (output_image, status_message)
            return output_image, status_message
            
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            return None, error_msg
    
    def _capture_sample(self, image, signal, gesture, hand_type):
        """Capture a detected umpire signal sample"""
        sample_rgb = cv.cvtColor(copy.deepcopy(image), cv.COLOR_BGR2RGB)
        self.captured_samples.append({
            'image': sample_rgb,
            'signal': signal,
            'gesture': gesture,
            'hand': hand_type,
            'frame': self.frame_count
        })
        print(f"Captured sample #{len(self.captured_samples)}: {signal}")
    
    def get_captured_samples(self):
        """Return captured samples for gallery display"""
        if len(self.captured_samples) == 0:
            return None
        return [
            (sample['image'], f"🏏 {sample['signal']}\n({sample['hand']} hand)")
            for sample in self.captured_samples
        ]
    
    def clear_samples(self):
        """Clear all captured samples"""
        count = len(self.captured_samples)
        self.captured_samples = []
        self.frame_count = 0
        self.last_detection_frame = 0
        self.current_detection = None
        self.consistent_detection_count = 0
        return count
    
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
    
    def _draw_landmarks(self, image, landmarks):
        """Draw hand landmarks"""
        if len(landmarks) == 0:
            return image
        
        # Draw connections
        connections = [
            # Thumb
            (0, 1), (1, 2), (2, 3), (3, 4),
            # Index
            (0, 5), (5, 6), (6, 7), (7, 8),
            # Middle
            (0, 9), (9, 10), (10, 11), (11, 12),
            # Ring
            (0, 13), (13, 14), (14, 15), (15, 16),
            # Pinky
            (0, 17), (17, 18), (18, 19), (19, 20),
            # Palm
            (5, 9), (9, 13), (13, 17)
        ]
        
        for connection in connections:
            start = landmarks[connection[0]]
            end = landmarks[connection[1]]
            cv.line(image, tuple(start), tuple(end), (255, 255, 255), 2)
        
        # Draw landmark points
        for i, landmark in enumerate(landmarks):
            if i in [4, 8, 12, 16, 20]:  # Fingertips
                cv.circle(image, tuple(landmark), 6, (0, 255, 0), -1)
                cv.circle(image, tuple(landmark), 6, (255, 255, 255), 1)
            else:
                cv.circle(image, tuple(landmark), 4, (0, 0, 255), -1)
                cv.circle(image, tuple(landmark), 4, (255, 255, 255), 1)
        
        return image
    
    def _draw_bounding_rect(self, image, brect):
        """Draw bounding rectangle"""
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                    (0, 255, 0), 2)
        return image
    
    def _draw_info_text(self, image, brect, handedness, signal, gesture):
        """Draw information text near hand"""
        hand_label = handedness.classification[0].label
        
        # Background for text
        cv.rectangle(image, (brect[0], brect[1] - 60), 
                    (brect[0] + 250, brect[1]), (0, 0, 0), -1)
        
        # Signal text
        cv.putText(image, f"Signal: {signal}", 
                  (brect[0] + 5, brect[1] - 35),
                  cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv.LINE_AA)
        
        # Hand and gesture
        cv.putText(image, f"{hand_label} - {gesture}", 
                  (brect[0] + 5, brect[1] - 10),
                  cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv.LINE_AA)
        
        return image
    
    def _draw_header(self, image, status):
        """Draw header with status"""
        h, w = image.shape[:2]
        
        # Header background
        cv.rectangle(image, (0, 0), (w, 60), (50, 50, 50), -1)
        
        # Title
        cv.putText(image, "Cricket Umpire Signs Detector", 
                  (10, 25),
                  cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv.LINE_AA)
        
        # Status
        cv.putText(image, status, 
                  (10, 50),
                  cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv.LINE_AA)
        
        # Sample count
        cv.putText(image, f"Samples: {len(self.captured_samples)}/{self.max_samples}", 
                  (w - 200, 25),
                  cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv.LINE_AA)
        
        return image


# Global detector instance
detector = None


def initialize_detector():
    global detector
    if detector is None:
        detector = CricketUmpireDetector()
    return detector


def process_video_frame(image):
    global detector
    if detector is None:
        detector = initialize_detector()
    
    if image is None:
        return None, "Waiting for camera..."
    
    return detector.process_frame(image)


def show_captured_samples():
    global detector
    if detector is None:
        return None
    samples = detector.get_captured_samples()
    if samples is None:
        return []
    return samples


def clear_samples():
    global detector
    if detector is not None:
        count = detector.clear_samples()
        return [], f"✓ Cleared {count} samples!"
    return [], "No samples to clear"


def create_interface():
    with gr.Blocks(title="Cricket Umpire Signs Detector", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🏏 Cricket Umpire Signs Detector
        ### Real-time detection of cricket umpire hand signals
        **Show umpire signals to the camera - they will be detected and saved automatically!**
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                webcam_input = gr.Image(
                    sources=["webcam"],
                    streaming=True,
                    label="📹 Camera Feed",
                    type="numpy",
                    mirror_webcam=False
                )
                status_box = gr.Textbox(
                    label="Detection Status",
                    value="Ready - Show umpire signals to start",
                    interactive=False,
                    lines=2
                )
                
                with gr.Row():
                    show_samples_btn = gr.Button("📊 Show Captured Samples", variant="primary")
                    clear_samples_btn = gr.Button("🗑️ Clear All Samples", variant="stop")
            
            with gr.Column(scale=1):
                output_display = gr.Image(
                    label="🎯 Detection Output",
                    type="numpy"
                )
                
                gr.Markdown("""
                ### 🏏 Umpire Signals Detected:
                - **Wide** - Open hand
                - **Out** - Closed fist
                - **No Ball** - Pointer finger
                - **Four** - OK sign
                - **Six** - Peace sign
                - And more...
                """)
        
        sample_gallery = gr.Gallery(
            label="🎬 Captured Umpire Signals",
            columns=3,
            rows=3,
            height="auto"
        )
        
        # Stream processing
        webcam_input.stream(
            fn=process_video_frame,
            inputs=webcam_input,
            outputs=[output_display, status_box],
            show_progress="hidden"
        )
        
        # Button actions
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
    print("=" * 60)
    print("Starting Cricket Umpire Signs Detector...")
    print("=" * 60)
    
    # Initialize detector
    initialize_detector()
    
    # Get port from environment or use default
    port = int(os.environ.get("PORT", 7860))
    
    # Create and launch interface
    demo = create_interface()
    demo.queue(max_size=20, default_concurrency_limit=5)
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False,
        show_error=True,
        debug=True
    )

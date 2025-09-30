# #!/usr/bin/env python
# # -*- coding: utf-8 -*-


# import csv
# import copy
# import argparse
# import itertools
# from collections import Counter
# from collections import deque

# import cv2 as cv
# import numpy as np
# import mediapipe as mp

# from utils import CvFpsCalc
# from model import KeyPointClassifier
# from model import PointHistoryClassifier


# def get_args():
#     parser = argparse.ArgumentParser()

#     parser.add_argument("--device", type=int, default=0)
#     parser.add_argument("--width", help='cap width', type=int, default=960)
#     parser.add_argument("--height", help='cap height', type=int, default=540)

#     parser.add_argument('--use_static_image_mode', action='store_true')
#     parser.add_argument("--min_detection_confidence",
#                         help='min_detection_confidence',
#                         type=float,
#                         default=0.7)
#     parser.add_argument("--min_tracking_confidence",
#                         help='min_tracking_confidence',
#                         type=int,
#                         default=0.5)

#     args = parser.parse_args()

#     return args


# def main():
#     # Argument parsing #################################################################
#     args = get_args()

#     cap_device = args.device
#     cap_width = args.width
#     cap_height = args.height

#     use_static_image_mode = args.use_static_image_mode
#     min_detection_confidence = args.min_detection_confidence
#     min_tracking_confidence = args.min_tracking_confidence

#     use_brect = True

#     # Camera preparation ###############################################################
#     cap = cv.VideoCapture(cap_device)
#     cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
#     cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

#     # Model load #############################################################
#     mp_hands = mp.solutions.hands
#     hands = mp_hands.Hands(
#         static_image_mode=use_static_image_mode,
#         max_num_hands=2,
#         min_detection_confidence=min_detection_confidence,
#         min_tracking_confidence=min_tracking_confidence,
#     )

#     keypoint_classifier = KeyPointClassifier()

#     point_history_classifier = PointHistoryClassifier()

#     # Read labels ###########################################################
#     with open('model/keypoint_classifier/keypoint_classifier_label.csv',
#               encoding='utf-8-sig') as f:
#         keypoint_classifier_labels = csv.reader(f)
#         keypoint_classifier_labels = [
#             row[0] for row in keypoint_classifier_labels
#         ]
#     with open(
#             'model/point_history_classifier/point_history_classifier_label.csv',
#             encoding='utf-8-sig') as f:
#         point_history_classifier_labels = csv.reader(f)
#         point_history_classifier_labels = [
#             row[0] for row in point_history_classifier_labels
#         ]

#     # FPS Measurement ########################################################
#     cvFpsCalc = CvFpsCalc(buffer_len=10)

#     # Coordinate history #################################################################
#     history_length = 16
#     point_history = deque(maxlen=history_length)

#     # Finger gesture history ################################################
#     finger_gesture_history = deque(maxlen=history_length)

#     #  ########################################################################
#     mode = 0

#     while True:
#         fps = cvFpsCalc.get()

#         # Process Key (ESC: end) #################################################
#         key = cv.waitKey(10)
#         if key == 27:  # ESC
#             break
#         number, mode = select_mode(key, mode)

#         # Camera capture #####################################################
#         ret, image = cap.read()
#         if not ret:
#             break
#         image = cv.flip(image, 1)  # Mirror display
#         debug_image = copy.deepcopy(image)

#         # Detection implementation #############################################################
#         image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

#         image.flags.writeable = False
#         results = hands.process(image)
#         image.flags.writeable = True

#         #  ####################################################################
#         if results.multi_hand_landmarks is not None:
#             for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
#                                                   results.multi_handedness):

#                 # Bounding box calculation
#                 brect = calc_bounding_rect(debug_image, hand_landmarks)
#                 # Landmark calculation
#                 landmark_list = calc_landmark_list(debug_image, hand_landmarks)

#                 # Conversion to relative coordinates / normalized coordinates
#                 pre_processed_landmark_list = pre_process_landmark(
#                     landmark_list)
#                 pre_processed_point_history_list = pre_process_point_history(
#                     debug_image, point_history)
#                 # Write to the dataset file
#                 logging_csv(number, mode, pre_processed_landmark_list,
#                             pre_processed_point_history_list)

#                 # Hand sign classification
#                 hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
#                 if hand_sign_id == "Not applicable":
#                     point_history.append(landmark_list[8])
#                 else:
#                     point_history.append([0, 0])

#                 # Finger gesture classification
#                 finger_gesture_id = 0
#                 point_history_len = len(pre_processed_point_history_list)
#                 if point_history_len == (history_length * 2):
#                     finger_gesture_id = point_history_classifier(
#                         pre_processed_point_history_list)

#                 # Calculates the gesture IDs in the latest detection
#                 finger_gesture_history.append(finger_gesture_id)
#                 most_common_fg_id = Counter(
#                     finger_gesture_history).most_common()

#                 # Drawing part
#                 debug_image = draw_bounding_rect(use_brect, debug_image, brect)
#                 debug_image = draw_landmarks(debug_image, landmark_list)
#                 debug_image = draw_info_text(
#                     debug_image,
#                     brect,
#                     handedness,
#                     keypoint_classifier_labels[hand_sign_id],
#                     point_history_classifier_labels[most_common_fg_id[0][0]],
#                 )
#         else:
#             point_history.append([0, 0])

#         debug_image = draw_point_history(debug_image, point_history)
#         debug_image = draw_info(debug_image, fps, mode, number)

#         # Screen reflection #############################################################
#         cv.imshow('Hand Gesture Recognition', debug_image)

#     cap.release()
#     cv.destroyAllWindows()


# def select_mode(key, mode):
#     number = -1
#     if 48 <= key <= 57:  # 0 ~ 9
#         number = key - 48
#     if key == 110:  # n
#         mode = 0
#     if key == 107:  # k
#         mode = 1
#     if key == 104:  # h
#         mode = 2
#     return number, mode


# def calc_bounding_rect(image, landmarks):
#     image_width, image_height = image.shape[1], image.shape[0]

#     landmark_array = np.empty((0, 2), int)

#     for _, landmark in enumerate(landmarks.landmark):
#         landmark_x = min(int(landmark.x * image_width), image_width - 1)
#         landmark_y = min(int(landmark.y * image_height), image_height - 1)

#         landmark_point = [np.array((landmark_x, landmark_y))]

#         landmark_array = np.append(landmark_array, landmark_point, axis=0)

#     x, y, w, h = cv.boundingRect(landmark_array)

#     return [x, y, x + w, y + h]


# def calc_landmark_list(image, landmarks):
#     image_width, image_height = image.shape[1], image.shape[0]

#     landmark_point = []

#     # Keypoint
#     for _, landmark in enumerate(landmarks.landmark):
#         landmark_x = min(int(landmark.x * image_width), image_width - 1)
#         landmark_y = min(int(landmark.y * image_height), image_height - 1)
#         # landmark_z = landmark.z

#         landmark_point.append([landmark_x, landmark_y])

#     return landmark_point


# def pre_process_landmark(landmark_list):
#     temp_landmark_list = copy.deepcopy(landmark_list)

#     # Convert to relative coordinates
#     base_x, base_y = 0, 0
#     for index, landmark_point in enumerate(temp_landmark_list):
#         if index == 0:
#             base_x, base_y = landmark_point[0], landmark_point[1]

#         temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
#         temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

#     # Convert to a one-dimensional list
#     temp_landmark_list = list(
#         itertools.chain.from_iterable(temp_landmark_list))

#     # Normalization
#     max_value = max(list(map(abs, temp_landmark_list)))

#     def normalize_(n):
#         return n / max_value

#     temp_landmark_list = list(map(normalize_, temp_landmark_list))

#     return temp_landmark_list


# def pre_process_point_history(image, point_history):
#     image_width, image_height = image.shape[1], image.shape[0]

#     temp_point_history = copy.deepcopy(point_history)

#     # Convert to relative coordinates
#     base_x, base_y = 0, 0
#     for index, point in enumerate(temp_point_history):
#         if index == 0:
#             base_x, base_y = point[0], point[1]

#         temp_point_history[index][0] = (temp_point_history[index][0] -
#                                         base_x) / image_width
#         temp_point_history[index][1] = (temp_point_history[index][1] -
#                                         base_y) / image_height

#     # Convert to a one-dimensional list
#     temp_point_history = list(
#         itertools.chain.from_iterable(temp_point_history))

#     return temp_point_history


# def logging_csv(number, mode, landmark_list, point_history_list):
#     if mode == 0:
#         pass
#     if mode == 1 and (0 <= number <= 9):
#         csv_path = 'model/keypoint_classifier/keypoint.csv'
#         with open(csv_path, 'a', newline="") as f:
#             writer = csv.writer(f)
#             writer.writerow([number, *landmark_list])
#     if mode == 2 and (0 <= number <= 9):
#         csv_path = 'model/point_history_classifier/point_history.csv'
#         with open(csv_path, 'a', newline="") as f:
#             writer = csv.writer(f)
#             writer.writerow([number, *point_history_list])
#     return


# def draw_landmarks(image, landmark_point):
#     if len(landmark_point) > 0:
#         # Thumb
#         cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
#                 (0, 0, 0), 6)
#         cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
#                 (255, 255, 255), 2)
#         cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
#                 (0, 0, 0), 6)
#         cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
#                 (255, 255, 255), 2)

#         # Index finger
#         cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
#                 (0, 0, 0), 6)
#         cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
#                 (255, 255, 255), 2)
#         cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
#                 (0, 0, 0), 6)
#         cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
#                 (255, 255, 255), 2)
#         cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
#                 (0, 0, 0), 6)
#         cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
#                 (255, 255, 255), 2)

#         # Middle finger
#         cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
#                 (0, 0, 0), 6)
#         cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
#                 (255, 255, 255), 2)
#         cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
#                 (0, 0, 0), 6)
#         cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
#                 (255, 255, 255), 2)
#         cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
#                 (0, 0, 0), 6)
#         cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
#                 (255, 255, 255), 2)

#         # Ring finger
#         cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
#                 (0, 0, 0), 6)
#         cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
#                 (255, 255, 255), 2)
#         cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
#                 (0, 0, 0), 6)
#         cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
#                 (255, 255, 255), 2)
#         cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
#                 (0, 0, 0), 6)
#         cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
#                 (255, 255, 255), 2)

#         # Little finger
#         cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
#                 (0, 0, 0), 6)
#         cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
#                 (255, 255, 255), 2)
#         cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
#                 (0, 0, 0), 6)
#         cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
#                 (255, 255, 255), 2)
#         cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
#                 (0, 0, 0), 6)
#         cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
#                 (255, 255, 255), 2)

#         # Palm
#         cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
#                 (0, 0, 0), 6)
#         cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
#                 (255, 255, 255), 2)
#         cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
#                 (0, 0, 0), 6)
#         cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
#                 (255, 255, 255), 2)
#         cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
#                 (0, 0, 0), 6)
#         cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
#                 (255, 255, 255), 2)
#         cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
#                 (0, 0, 0), 6)
#         cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
#                 (255, 255, 255), 2)
#         cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
#                 (0, 0, 0), 6)
#         cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
#                 (255, 255, 255), 2)
#         cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
#                 (0, 0, 0), 6)
#         cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
#                 (255, 255, 255), 2)
#         cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
#                 (0, 0, 0), 6)
#         cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
#                 (255, 255, 255), 2)

#     # Key Points
#     for index, landmark in enumerate(landmark_point):
#         if index == 0:  # 手首1
#             cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
#                       -1)
#             cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
#         if index == 1:  # 手首2
#             cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
#                       -1)
#             cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
#         if index == 2:  # 親指：付け根
#             cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
#                       -1)
#             cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
#         if index == 3:  # 親指：第1関節
#             cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
#                       -1)
#             cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
#         if index == 4:  # 親指：指先
#             cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
#                       -1)
#             cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
#         if index == 5:  # 人差指：付け根
#             cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
#                       -1)
#             cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
#         if index == 6:  # 人差指：第2関節
#             cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
#                       -1)
#             cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
#         if index == 7:  # 人差指：第1関節
#             cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
#                       -1)
#             cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
#         if index == 8:  # 人差指：指先
#             cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
#                       -1)
#             cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
#         if index == 9:  # 中指：付け根
#             cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
#                       -1)
#             cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
#         if index == 10:  # 中指：第2関節
#             cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
#                       -1)
#             cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
#         if index == 11:  # 中指：第1関節
#             cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
#                       -1)
#             cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
#         if index == 12:  # 中指：指先
#             cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
#                       -1)
#             cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
#         if index == 13:  # 薬指：付け根
#             cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
#                       -1)
#             cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
#         if index == 14:  # 薬指：第2関節
#             cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
#                       -1)
#             cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
#         if index == 15:  # 薬指：第1関節
#             cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
#                       -1)
#             cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
#         if index == 16:  # 薬指：指先
#             cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
#                       -1)
#             cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
#         if index == 17:  # 小指：付け根
#             cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
#                       -1)
#             cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
#         if index == 18:  # 小指：第2関節
#             cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
#                       -1)
#             cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
#         if index == 19:  # 小指：第1関節
#             cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
#                       -1)
#             cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
#         if index == 20:  # 小指：指先
#             cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
#                       -1)
#             cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)

#     return image


# def draw_bounding_rect(use_brect, image, brect):
#     if use_brect:
#         # Outer rectangle
#         cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
#                      (0, 0, 0), 1)

#     return image


# def draw_info_text(image, brect, handedness, hand_sign_text,
#                    finger_gesture_text):
#     cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22),
#                  (0, 0, 0), -1)

#     info_text = handedness.classification[0].label[0:]
#     if hand_sign_text != "":
#         info_text = info_text + ':' + hand_sign_text
#     cv.putText(image, info_text, (brect[0] + 5, brect[1] - 4),
#                cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)

#     if finger_gesture_text != "":
#         cv.putText(image, "Finger Gesture:" + finger_gesture_text, (10, 60),
#                    cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv.LINE_AA)
#         cv.putText(image, "Finger Gesture:" + finger_gesture_text, (10, 60),
#                    cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2,
#                    cv.LINE_AA)

#     return image


# def draw_point_history(image, point_history):
#     for index, point in enumerate(point_history):
#         if point[0] != 0 and point[1] != 0:
#             cv.circle(image, (point[0], point[1]), 1 + int(index / 2),
#                       (152, 251, 152), 2)

#     return image


# def draw_info(image, fps, mode, number):
#     cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
#                1.0, (0, 0, 0), 4, cv.LINE_AA)
#     cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
#                1.0, (255, 255, 255), 2, cv.LINE_AA)

#     mode_string = ['Logging Key Point', 'Logging Point History']
#     if 1 <= mode <= 2:
#         cv.putText(image, "MODE:" + mode_string[mode - 1], (10, 90),
#                    cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
#                    cv.LINE_AA)
#         if 0 <= number <= 9:
#             cv.putText(image, "NUM:" + str(number), (10, 110),
#                        cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
#                        cv.LINE_AA)
#     return image


# if __name__ == '__main__':
#     main()
# # !/usr/bin/env python
# # -*- coding: utf-8 -*-

# import csv
# import copy
# import itertools
# from collections import Counter
# from collections import deque
# import cv2 as cv
# import numpy as np
# import mediapipe as mp
# import gradio as gr
# from model import KeyPointClassifier
# from model import PointHistoryClassifier


# class HandGestureRecognizer:
#     def __init__(self):
#         # Model load
#         mp_hands = mp.solutions.hands
#         self.hands = mp_hands.Hands(
#             static_image_mode=False,
#             max_num_hands=2,
#             min_detection_confidence=0.7,
#             min_tracking_confidence=0.5,
#         )

#         self.keypoint_classifier = KeyPointClassifier()
#         self.point_history_classifier = PointHistoryClassifier()

#         # Read labels
#         with open('model/keypoint_classifier/keypoint_classifier_label.csv',
#                   encoding='utf-8-sig') as f:
#             keypoint_classifier_labels = csv.reader(f)
#             self.keypoint_classifier_labels = [
#                 row[0] for row in keypoint_classifier_labels
#             ]

#         with open('model/point_history_classifier/point_history_classifier_label.csv',
#                   encoding='utf-8-sig') as f:
#             point_history_classifier_labels = csv.reader(f)
#             self.point_history_classifier_labels = [
#                 row[0] for row in point_history_classifier_labels
#             ]

#         # Coordinate history
#         self.history_length = 16
#         self.point_history = deque(maxlen=self.history_length)
#         self.finger_gesture_history = deque(maxlen=self.history_length)
        
#         # Sample capture storage
#         self.captured_samples = []
#         self.max_samples = 10  # Maximum number of samples to store

#     def process_frame(self, image):
#         if image is None:
#             return None

#         # Mirror display
#         image = cv.flip(image, 1)
#         debug_image = copy.deepcopy(image)

#         # Detection implementation
#         image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
#         results = self.hands.process(image_rgb)

#         if results.multi_hand_landmarks is not None:
#             for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
#                                                   results.multi_handedness):
#                 # Bounding box calculation
#                 brect = self.calc_bounding_rect(debug_image, hand_landmarks)

#                 # Landmark calculation
#                 landmark_list = self.calc_landmark_list(debug_image, hand_landmarks)

#                 # Conversion to relative coordinates / normalized coordinates
#                 pre_processed_landmark_list = self.pre_process_landmark(landmark_list)
#                 pre_processed_point_history_list = self.pre_process_point_history(
#                     debug_image, self.point_history)

#                 # Hand sign classification
#                 hand_sign_id = self.keypoint_classifier(pre_processed_landmark_list)
#                 if hand_sign_id == 2:  # Assuming 2 is "Point" gesture
#                     self.point_history.append(landmark_list[8])
#                 else:
#                     self.point_history.append([0, 0])

#                 # Finger gesture classification
#                 finger_gesture_id = 0
#                 point_history_len = len(pre_processed_point_history_list)
#                 if point_history_len == (self.history_length * 2):
#                     finger_gesture_id = self.point_history_classifier(
#                         pre_processed_point_history_list)

#                 # Calculates the gesture IDs in the latest detection
#                 self.finger_gesture_history.append(finger_gesture_id)
#                 most_common_fg_id = Counter(self.finger_gesture_history).most_common()

#                 # Drawing part
#                 debug_image = self.draw_bounding_rect(True, debug_image, brect)
#                 debug_image = self.draw_landmarks(debug_image, landmark_list)
#                 debug_image = self.draw_info_text(
#                     debug_image,
#                     brect,
#                     handedness,
#                     self.keypoint_classifier_labels[hand_sign_id],
#                     self.point_history_classifier_labels[most_common_fg_id[0][0]],
#                 )
                
#                 # Capture sample every 30 frames (approximately 1 per second at 30fps)
#                 if len(self.captured_samples) < self.max_samples:
#                     if len(self.captured_samples) == 0 or np.random.random() < 0.03:  # ~1 per second
#                         sample_info = {
#                             'image': copy.deepcopy(debug_image),
#                             'gesture': self.keypoint_classifier_labels[hand_sign_id],
#                             'hand': handedness.classification[0].label
#                         }
#                         self.captured_samples.append(sample_info)
#         else:
#             self.point_history.append([0, 0])

#         debug_image = self.draw_point_history(debug_image, self.point_history)

#         return debug_image

#     def get_captured_samples(self):
#         """Return all captured sample images"""
#         return [sample['image'] for sample in self.captured_samples]
    
#     def clear_samples(self):
#         """Clear all captured samples"""
#         self.captured_samples = []

#     def calc_bounding_rect(self, image, landmarks):
#         image_width, image_height = image.shape[1], image.shape[0]
#         landmark_array = np.empty((0, 2), int)

#         for _, landmark in enumerate(landmarks.landmark):
#             landmark_x = min(int(landmark.x * image_width), image_width - 1)
#             landmark_y = min(int(landmark.y * image_height), image_height - 1)
#             landmark_point = [np.array((landmark_x, landmark_y))]
#             landmark_array = np.append(landmark_array, landmark_point, axis=0)

#         x, y, w, h = cv.boundingRect(landmark_array)
#         return [x, y, x + w, y + h]

#     def calc_landmark_list(self, image, landmarks):
#         image_width, image_height = image.shape[1], image.shape[0]
#         landmark_point = []

#         for _, landmark in enumerate(landmarks.landmark):
#             landmark_x = min(int(landmark.x * image_width), image_width - 1)
#             landmark_y = min(int(landmark.y * image_height), image_height - 1)
#             landmark_point.append([landmark_x, landmark_y])

#         return landmark_point

#     def pre_process_landmark(self, landmark_list):
#         temp_landmark_list = copy.deepcopy(landmark_list)

#         # Convert to relative coordinates
#         base_x, base_y = 0, 0
#         for index, landmark_point in enumerate(temp_landmark_list):
#             if index == 0:
#                 base_x, base_y = landmark_point[0], landmark_point[1]

#             temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
#             temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

#         # Convert to a one-dimensional list
#         temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))

#         # Normalization
#         max_value = max(list(map(abs, temp_landmark_list)))

#         def normalize_(n):
#             return n / max_value

#         temp_landmark_list = list(map(normalize_, temp_landmark_list))
#         return temp_landmark_list

#     def pre_process_point_history(self, image, point_history):
#         image_width, image_height = image.shape[1], image.shape[0]
#         temp_point_history = copy.deepcopy(point_history)

#         # Convert to relative coordinates
#         base_x, base_y = 0, 0
#         for index, point in enumerate(temp_point_history):
#             if index == 0:
#                 base_x, base_y = point[0], point[1]

#             temp_point_history[index][0] = (temp_point_history[index][0] - base_x) / image_width
#             temp_point_history[index][1] = (temp_point_history[index][1] - base_y) / image_height

#         # Convert to a one-dimensional list
#         temp_point_history = list(itertools.chain.from_iterable(temp_point_history))
#         return temp_point_history

#     def draw_landmarks(self, image, landmark_point):
#         if len(landmark_point) > 0:
#             # Thumb
#             cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]), (0, 0, 0), 6)
#             cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]), (255, 255, 255), 2)
#             cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]), (0, 0, 0), 6)
#             cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]), (255, 255, 255), 2)

#             # Index finger
#             cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]), (0, 0, 0), 6)
#             cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]), (255, 255, 255), 2)
#             cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]), (0, 0, 0), 6)
#             cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]), (255, 255, 255), 2)
#             cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]), (0, 0, 0), 6)
#             cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]), (255, 255, 255), 2)

#             # Middle finger
#             cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]), (0, 0, 0), 6)
#             cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]), (255, 255, 255), 2)
#             cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]), (0, 0, 0), 6)
#             cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]), (255, 255, 255), 2)
#             cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]), (0, 0, 0), 6)
#             cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]), (255, 255, 255), 2)

#             # Ring finger
#             cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]), (0, 0, 0), 6)
#             cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]), (255, 255, 255), 2)
#             cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]), (0, 0, 0), 6)
#             cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]), (255, 255, 255), 2)
#             cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]), (0, 0, 0), 6)
#             cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]), (255, 255, 255), 2)

#             # Little finger
#             cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]), (0, 0, 0), 6)
#             cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]), (255, 255, 255), 2)
#             cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]), (0, 0, 0), 6)
#             cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]), (255, 255, 255), 2)
#             cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]), (0, 0, 0), 6)
#             cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]), (255, 255, 255), 2)

#             # Palm
#             cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]), (0, 0, 0), 6)
#             cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]), (255, 255, 255), 2)
#             cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]), (0, 0, 0), 6)
#             cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]), (255, 255, 255), 2)
#             cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]), (0, 0, 0), 6)
#             cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]), (255, 255, 255), 2)
#             cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]), (0, 0, 0), 6)
#             cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]), (255, 255, 255), 2)
#             cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]), (0, 0, 0), 6)
#             cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]), (255, 255, 255), 2)
#             cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]), (0, 0, 0), 6)
#             cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]), (255, 255, 255), 2)
#             cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]), (0, 0, 0), 6)
#             cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]), (255, 255, 255), 2)

#         # Key Points
#         for index, landmark in enumerate(landmark_point):
#             if index in [0, 1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19]:
#                 cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
#                 cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
#             if index in [4, 8, 12, 16, 20]:  # Fingertips
#                 cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255), -1)
#                 cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)

#         return image

#     def draw_bounding_rect(self, use_brect, image, brect):
#         if use_brect:
#             cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]), (0, 0, 0), 1)
#         return image

#     def draw_info_text(self, image, brect, handedness, hand_sign_text, finger_gesture_text):
#         cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22), (0, 0, 0), -1)

#         info_text = handedness.classification[0].label[0:]
#         if hand_sign_text != "":
#             info_text = info_text + ':' + hand_sign_text
#         cv.putText(image, info_text, (brect[0] + 5, brect[1] - 4),
#                    cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)

#         if finger_gesture_text != "":
#             cv.putText(image, "Finger Gesture:" + finger_gesture_text, (10, 60),
#                        cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv.LINE_AA)
#             cv.putText(image, "Finger Gesture:" + finger_gesture_text, (10, 60),
#                        cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv.LINE_AA)

#         return image

#     def draw_point_history(self, image, point_history):
#         for index, point in enumerate(point_history):
#             if point[0] != 0 and point[1] != 0:
#                 cv.circle(image, (point[0], point[1]), 1 + int(index / 2), (152, 251, 152), 2)
#         return image


# # Initialize the recognizer
# recognizer = HandGestureRecognizer()


# def predict(image):
#     """Process each frame from the webcam"""
#     if image is None:
#         return None
#     return recognizer.process_frame(image)


# def show_samples():
#     """Display captured sample images"""
#     samples = recognizer.get_captured_samples()
#     if len(samples) == 0:
#         return None
#     return samples


# def clear_and_restart():
#     """Clear captured samples for a new session"""
#     recognizer.clear_samples()
#     return None


# # Create Gradio interface with sample display
# with gr.Blocks() as demo:
#     gr.Markdown("# Hand Gesture Recognition with Sample Capture")
#     gr.Markdown("Allow camera access to detect hand gestures in real-time. Samples are automatically captured during detection.")
    
#     with gr.Row():
#         with gr.Column():
#             webcam_input = gr.Image(sources=["webcam"], streaming=True, label="Live Feed")
#             webcam_output = gr.Image(label="Detection Output")
            
#     with gr.Row():
#         show_btn = gr.Button("Show Captured Samples", variant="primary")
#         clear_btn = gr.Button("Clear Samples & Restart")
    
#     sample_gallery = gr.Gallery(label="Captured Detection Samples", columns=3, rows=2)
    
#     # Set up streaming
#     webcam_input.stream(
#         fn=predict,
#         inputs=webcam_input,
#         outputs=webcam_output,
#     )
    
#     # Button actions
#     show_btn.click(fn=show_samples, outputs=sample_gallery)
#     clear_btn.click(fn=clear_and_restart, outputs=sample_gallery)

# if __name__ == "__main__":
#     demo.launch(
#         server_name="0.0.0.0",
#         server_port=10000,
#         share=False
#     )
#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import csv
import copy
import itertools
from collections import Counter
from collections import deque
import cv2 as cv
import numpy as np
import mediapipe as mp
import gradio as gr
from model import KeyPointClassifier
from model import PointHistoryClassifier


class HandGestureRecognizer:
    def __init__(self):
        # Model load
        mp_hands = mp.solutions.hands
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
        )

        self.keypoint_classifier = KeyPointClassifier()
        self.point_history_classifier = PointHistoryClassifier()

        # Read labels
        with open('model/keypoint_classifier/keypoint_classifier_label.csv',
                  encoding='utf-8-sig') as f:
            keypoint_classifier_labels = csv.reader(f)
            self.keypoint_classifier_labels = [
                row[0] for row in keypoint_classifier_labels
            ]

        with open('model/point_history_classifier/point_history_classifier_label.csv',
                  encoding='utf-8-sig') as f:
            point_history_classifier_labels = csv.reader(f)
            self.point_history_classifier_labels = [
                row[0] for row in point_history_classifier_labels
            ]

        # Coordinate history
        self.history_length = 16
        self.point_history = deque(maxlen=self.history_length)
        self.finger_gesture_history = deque(maxlen=self.history_length)
        
        # Sample capture storage
        self.captured_samples = []
        self.max_samples = 10

    def process_frame(self, image):
        if image is None:
            return None

        # Mirror display
        image = cv.flip(image, 1)
        debug_image = copy.deepcopy(image)

        # Detection implementation
        image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)

        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                  results.multi_handedness):
                # Bounding box calculation
                brect = self.calc_bounding_rect(debug_image, hand_landmarks)

                # Landmark calculation
                landmark_list = self.calc_landmark_list(debug_image, hand_landmarks)

                # Conversion to relative coordinates / normalized coordinates
                pre_processed_landmark_list = self.pre_process_landmark(landmark_list)
                pre_processed_point_history_list = self.pre_process_point_history(
                    debug_image, self.point_history)

                # Hand sign classification
                hand_sign_id = self.keypoint_classifier(pre_processed_landmark_list)
                if hand_sign_id == 2:
                    self.point_history.append(landmark_list[8])
                else:
                    self.point_history.append([0, 0])

                # Finger gesture classification
                finger_gesture_id = 0
                point_history_len = len(pre_processed_point_history_list)
                if point_history_len == (self.history_length * 2):
                    finger_gesture_id = self.point_history_classifier(
                        pre_processed_point_history_list)

                # Calculates the gesture IDs in the latest detection
                self.finger_gesture_history.append(finger_gesture_id)
                most_common_fg_id = Counter(self.finger_gesture_history).most_common()

                # Drawing part
                debug_image = self.draw_bounding_rect(True, debug_image, brect)
                debug_image = self.draw_landmarks(debug_image, landmark_list)
                debug_image = self.draw_info_text(
                    debug_image,
                    brect,
                    handedness,
                    self.keypoint_classifier_labels[hand_sign_id],
                    self.point_history_classifier_labels[most_common_fg_id[0][0]],
                )
                
                # Capture sample
                if len(self.captured_samples) < self.max_samples:
                    if len(self.captured_samples) == 0 or np.random.random() < 0.03:
                        sample_info = {
                            'image': copy.deepcopy(debug_image),
                            'gesture': self.keypoint_classifier_labels[hand_sign_id],
                            'hand': handedness.classification[0].label
                        }
                        self.captured_samples.append(sample_info)
        else:
            self.point_history.append([0, 0])

        debug_image = self.draw_point_history(debug_image, self.point_history)
        return debug_image

    def get_captured_samples(self):
        """Return all captured sample images"""
        return [sample['image'] for sample in self.captured_samples]
    
    def clear_samples(self):
        """Clear all captured samples"""
        self.captured_samples = []

    def calc_bounding_rect(self, image, landmarks):
        image_width, image_height = image.shape[1], image.shape[0]
        landmark_array = np.empty((0, 2), int)

        for _, landmark in enumerate(landmarks.landmark):
            landmark_x = min(int(landmark.x * image_width), image_width - 1)
            landmark_y = min(int(landmark.y * image_height), image_height - 1)
            landmark_point = [np.array((landmark_x, landmark_y))]
            landmark_array = np.append(landmark_array, landmark_point, axis=0)

        x, y, w, h = cv.boundingRect(landmark_array)
        return [x, y, x + w, y + h]

    def calc_landmark_list(self, image, landmarks):
        image_width, image_height = image.shape[1], image.shape[0]
        landmark_point = []

        for _, landmark in enumerate(landmarks.landmark):
            landmark_x = min(int(landmark.x * image_width), image_width - 1)
            landmark_y = min(int(landmark.y * image_height), image_height - 1)
            landmark_point.append([landmark_x, landmark_y])

        return landmark_point

    def pre_process_landmark(self, landmark_list):
        temp_landmark_list = copy.deepcopy(landmark_list)

        base_x, base_y = 0, 0
        for index, landmark_point in enumerate(temp_landmark_list):
            if index == 0:
                base_x, base_y = landmark_point[0], landmark_point[1]

            temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
            temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

        temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))
        max_value = max(list(map(abs, temp_landmark_list)))

        def normalize_(n):
            return n / max_value

        temp_landmark_list = list(map(normalize_, temp_landmark_list))
        return temp_landmark_list

    def pre_process_point_history(self, image, point_history):
        image_width, image_height = image.shape[1], image.shape[0]
        temp_point_history = copy.deepcopy(point_history)

        base_x, base_y = 0, 0
        for index, point in enumerate(temp_point_history):
            if index == 0:
                base_x, base_y = point[0], point[1]

            temp_point_history[index][0] = (temp_point_history[index][0] - base_x) / image_width
            temp_point_history[index][1] = (temp_point_history[index][1] - base_y) / image_height

        temp_point_history = list(itertools.chain.from_iterable(temp_point_history))
        return temp_point_history

    def draw_landmarks(self, image, landmark_point):
        if len(landmark_point) > 0:
            # Thumb
            cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]), (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]), (255, 255, 255), 2)
            cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]), (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]), (255, 255, 255), 2)

            # Index finger
            cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]), (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]), (255, 255, 255), 2)
            cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]), (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]), (255, 255, 255), 2)
            cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]), (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]), (255, 255, 255), 2)

            # Middle finger
            cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]), (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]), (255, 255, 255), 2)
            cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]), (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]), (255, 255, 255), 2)
            cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]), (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]), (255, 255, 255), 2)

            # Ring finger
            cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]), (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]), (255, 255, 255), 2)
            cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]), (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]), (255, 255, 255), 2)
            cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]), (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]), (255, 255, 255), 2)

            # Little finger
            cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]), (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]), (255, 255, 255), 2)
            cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]), (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]), (255, 255, 255), 2)
            cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]), (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]), (255, 255, 255), 2)

            # Palm
            cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]), (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]), (255, 255, 255), 2)
            cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]), (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]), (255, 255, 255), 2)
            cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]), (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]), (255, 255, 255), 2)
            cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]), (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]), (255, 255, 255), 2)
            cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]), (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]), (255, 255, 255), 2)
            cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]), (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]), (255, 255, 255), 2)
            cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]), (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]), (255, 255, 255), 2)

        for index, landmark in enumerate(landmark_point):
            if index in [0, 1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19]:
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index in [4, 8, 12, 16, 20]:
                cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255), -1)
                cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)

        return image

    def draw_bounding_rect(self, use_brect, image, brect):
        if use_brect:
            cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]), (0, 0, 0), 1)
        return image

    def draw_info_text(self, image, brect, handedness, hand_sign_text, finger_gesture_text):
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22), (0, 0, 0), -1)

        info_text = handedness.classification[0].label[0:]
        if hand_sign_text != "":
            info_text = info_text + ':' + hand_sign_text
        cv.putText(image, info_text, (brect[0] + 5, brect[1] - 4),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)

        if finger_gesture_text != "":
            cv.putText(image, "Finger Gesture:" + finger_gesture_text, (10, 60),
                       cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv.LINE_AA)
            cv.putText(image, "Finger Gesture:" + finger_gesture_text, (10, 60),
                       cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv.LINE_AA)

        return image

    def draw_point_history(self, image, point_history):
        for index, point in enumerate(point_history):
            if point[0] != 0 and point[1] != 0:
                cv.circle(image, (point[0], point[1]), 1 + int(index / 2), (152, 251, 152), 2)
        return image


# Initialize the recognizer
recognizer = HandGestureRecognizer()


def predict(image):
    """Process each frame from the webcam"""
    if image is None:
        return None
    return recognizer.process_frame(image)


def show_samples():
    """Display captured sample images"""
    samples = recognizer.get_captured_samples()
    if len(samples) == 0:
        return []
    return samples


def clear_and_restart():
    """Clear captured samples for a new session"""
    recognizer.clear_samples()
    return []


# Create Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# Hand Gesture Recognition with Sample Capture")
    gr.Markdown("Allow camera access to detect hand gestures in real-time. Samples are automatically captured during detection.")
    
    with gr.Row():
        with gr.Column():
            webcam_input = gr.Image(sources=["webcam"], streaming=True, label="Live Feed")
            webcam_output = gr.Image(label="Detection Output")
            
    with gr.Row():
        show_btn = gr.Button("Show Captured Samples", variant="primary")
        clear_btn = gr.Button("Clear Samples & Restart")
    
    sample_gallery = gr.Gallery(label="Captured Detection Samples", columns=3, rows=2)
    
    # Set up streaming
    webcam_input.stream(
        fn=predict,
        inputs=webcam_input,
        outputs=webcam_output,
    )
    
    # Button actions
    show_btn.click(fn=show_samples, outputs=sample_gallery)
    clear_btn.click(fn=clear_and_restart, outputs=sample_gallery)


if __name__ == "__main__":
    # Get port from environment variable or default to 10000
    port = int(os.environ.get("PORT", 10000))
    
    # Check if running in production environment
    is_production = os.environ.get('RENDER') is not None or os.environ.get('SPACE_ID') is not None
    
    if is_production:
        # Production settings for Render or Hugging Face Spaces
        demo.queue()  # Enable queue for better handling
        demo.launch(
            server_name="0.0.0.0",
            server_port=port,
            share=False,
            show_error=True
        )
    else:
        # Local development settings
        demo.launch(share=False)



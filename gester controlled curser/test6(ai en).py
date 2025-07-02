import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time
from datetime import datetime

# Kalman Filter class
class Kalman:
    def __init__(self):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]], np.float32)
        self.kf.transitionMatrix = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03

    def predict(self, x, y):
        prediction = self.kf.predict()
        measurement = np.array([[np.float32(x)], [np.float32(y)]])
        estimate = self.kf.correct(measurement)
        return int(estimate[0]), int(estimate[1])

# Initialize
cap = cv2.VideoCapture(0)
hand_detector = mp.solutions.hands.Hands(max_num_hands=1)
drawing_utils = mp.solutions.drawing_utils
screen_w, screen_h = pyautogui.size()

clicking = False
dragging = False
right_click_ready = True
kalman = Kalman()

# Logging function
def log_gesture(name):
    with open("gesture_log.txt", "a") as f:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"[{timestamp}] {name}\n")

while True:
    success, frame = cap.read()
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hand_detector.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            drawing_utils.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
            lm = hand_landmarks.landmark

            # Get finger tip coordinates
            x_index = int(lm[8].x * screen_w)
            y_index = int(lm[8].y * screen_h)
            x_thumb = int(lm[4].x * screen_w)
            y_thumb = int(lm[4].y * screen_h)

            # Smooth movement using Kalman filter
            smoothed_x, smoothed_y = kalman.predict(x_index, y_index)
            pyautogui.moveTo(smoothed_x, smoothed_y)

            # Finger up/down detection
            def is_finger_up(tip, pip):
                return lm[tip].y < lm[pip].y

            index_up = is_finger_up(8, 6)
            middle_up = is_finger_up(12, 10)
            ring_up = is_finger_up(16, 14)

            # Pinch distance
            pinch_distance = np.hypot(x_index - x_thumb, y_index - y_thumb)

            # --- Left Click ---
            if pinch_distance < 30 and not clicking:
                pyautogui.click()
                log_gesture("Left Click")
                clicking = True
                cv2.putText(frame, "Left Click", (x_index+20, y_index-30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            if pinch_distance > 40:
                clicking = False

            # --- Scroll ---
            if index_up and middle_up and not ring_up:
                pyautogui.scroll(-20)
                log_gesture("Scroll")
                cv2.putText(frame, "Scroll", (x_index+20, y_index-30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

            # --- Drag ---
            if pinch_distance < 30:
                if not dragging:
                    pyautogui.mouseDown()
                    log_gesture("Drag Start")
                    dragging = True
                cv2.putText(frame, "Dragging", (x_index+20, y_index-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)
            else:
                if dragging:
                    pyautogui.mouseUp()
                    log_gesture("Drag End")
                    dragging = False

            # --- Right Click ---
            if index_up and middle_up and ring_up:
                if right_click_ready:
                    pyautogui.rightClick()
                    log_gesture("Right Click")
                    right_click_ready = False
                    cv2.putText(frame, "Right Click", (x_index+20, y_index-90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            else:
                right_click_ready = True

    cv2.imshow("AI Gesture Mouse Control", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

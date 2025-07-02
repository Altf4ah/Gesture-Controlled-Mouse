import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time

# Setup
cap = cv2.VideoCapture(0)
hand_detector = mp.solutions.hands.Hands(max_num_hands=1)
drawing_utils = mp.solutions.drawing_utils
screen_w, screen_h = pyautogui.size()
prev_x, prev_y = 0, 0
smoothing = 7
clicking = False
dragging = False
drag_start_time = 0

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
            x_middle = int(lm[12].x * screen_w)
            y_middle = int(lm[12].y * screen_h)
            x_thumb = int(lm[4].x * screen_w)
            y_thumb = int(lm[4].y * screen_h)

            # Cursor movement with smoothing
            curr_x = prev_x + (x_index - prev_x) / smoothing
            curr_y = prev_y + (y_index - prev_y) / smoothing
            pyautogui.moveTo(curr_x, curr_y)
            prev_x, prev_y = curr_x, curr_y

            # Distance between index and thumb
            pinch_distance = np.hypot(x_index - x_thumb, y_index - y_thumb)

            # Check fingers up
            fingers_up = [lm[8].y < lm[6].y, lm[12].y < lm[10].y]  # index and middle
            # --- Click ---
            if pinch_distance < 30 and not clicking:
                pyautogui.click()
                clicking = True
                cv2.putText(frame, "Click", (x_index+20, y_index-30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            if pinch_distance > 40:
                clicking = False

            # --- Scroll ---
            if fingers_up[0] and fingers_up[1]:
                pyautogui.scroll(-20)
                cv2.putText(frame, "Scroll", (x_index+20, y_index-30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

            # --- Drag ---
            if pinch_distance < 30:
                if not dragging:
                    pyautogui.mouseDown()
                    dragging = True
                    drag_start_time = time.time()
                cv2.putText(frame, "Dragging", (x_index+20, y_index-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            else:
                if dragging:
                    pyautogui.mouseUp()
                    dragging = False

    cv2.imshow("Gesture Control", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

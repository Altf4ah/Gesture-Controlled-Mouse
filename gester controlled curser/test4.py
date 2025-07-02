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
right_click_ready = True

while True:
    success, frame = cap.read()
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hand_detector.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            drawing_utils.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
            lm = hand_landmarks.landmark

            # Get positions
            x_index = int(lm[8].x * screen_w)
            y_index = int(lm[8].y * screen_h)
            x_thumb = int(lm[4].x * screen_w)
            y_thumb = int(lm[4].y * screen_h)

            # Cursor movement
            curr_x = prev_x + (x_index - prev_x) / smoothing
            curr_y = prev_y + (y_index - prev_y) / smoothing
            pyautogui.moveTo(curr_x, curr_y)
            prev_x, prev_y = curr_x, curr_y

            # Finger status: up or not
            def is_finger_up(tip, pip):
                return lm[tip].y < lm[pip].y

            index_up = is_finger_up(8, 6)
            middle_up = is_finger_up(12, 10)
            ring_up = is_finger_up(16, 14)

            # --- Left Click (Pinch) ---
            pinch_distance = np.hypot(x_index - x_thumb, y_index - y_thumb)
            if pinch_distance < 30 and not clicking:
                pyautogui.click()
                clicking = True
                cv2.putText(frame, "Left Click", (x_index+20, y_index-30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            if pinch_distance > 40:
                clicking = False

            # --- Scroll (Index + Middle) ---
            if index_up and middle_up and not ring_up:
                pyautogui.scroll(-20)
                cv2.putText(frame, "Scroll", (x_index+20, y_index-30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

            # --- Drag (Pinch and Hold) ---
            if pinch_distance < 30:
                if not dragging:
                    pyautogui.mouseDown()
                    dragging = True
                cv2.putText(frame, "Dragging", (x_index+20, y_index-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            else:
                if dragging:
                    pyautogui.mouseUp()
                    dragging = False

            # --- Right Click (3 fingers: index + middle + ring) ---
            if index_up and middle_up and ring_up:
                if right_click_ready:
                    pyautogui.rightClick()
                    right_click_ready = False
                    cv2.putText(frame, "Right Click", (x_index+20, y_index-90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            else:
                right_click_ready = True

    cv2.imshow("Hand Gesture Control", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

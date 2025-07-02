import cv2
import mediapipe as mp
import pyautogui
import numpy as np

# Initialize modules
cap = cv2.VideoCapture(0)
hand_detector = mp.solutions.hands.Hands(max_num_hands=1)
drawing_utils = mp.solutions.drawing_utils
screen_w, screen_h = pyautogui.size()

# Store previous cursor position for smoothing
prev_x, prev_y = 0, 0
smoothing = 7

while True:
    success, frame = cap.read()
    frame = cv2.flip(frame, 1)  # Mirror image
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hand_detector.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks on frame
            drawing_utils.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)

            # Get index finger tip (landmark 8) and thumb tip (landmark 4)
            lm_list = hand_landmarks.landmark
            x_index = int(lm_list[8].x * screen_w)
            y_index = int(lm_list[8].y * screen_h)

            x_thumb = int(lm_list[4].x * screen_w)
            y_thumb = int(lm_list[4].y * screen_h)

            # Smoothing the cursor movement
            curr_x = prev_x + (x_index - prev_x) / smoothing
            curr_y = prev_y + (y_index - prev_y) / smoothing
            pyautogui.moveTo(curr_x, curr_y)
            prev_x, prev_y = curr_x, curr_y

            # Check if index and thumb are close (for clicking)
            distance = np.hypot(x_thumb - x_index, y_thumb - y_index)
            if distance < 30:
                pyautogui.click()
                cv2.putText(frame, "Click", (x_index + 20, y_index - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Hand Gesture Mouse", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

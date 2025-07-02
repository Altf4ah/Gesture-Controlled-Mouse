import cv2
import mediapipe as mp
import pyautogui
import time
import math

# Initialize modules
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1)

# Screen size
screen_w, screen_h = pyautogui.size()

# Webcam
cap = cv2.VideoCapture(0)

# To prevent multiple clicks
blink_click_cooldown = 1  # in seconds
last_click_time = 0

def euclidean_dist(p1, p2):
    return math.hypot(p1.x - p2.x, p1.y - p2.y)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    img_h, img_w, _ = frame.shape

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark

        # Cursor movement using right iris
        right_iris = landmarks[474]
        cursor_x = screen_w * right_iris.x
        cursor_y = screen_h * right_iris.y
        pyautogui.moveTo(cursor_x, cursor_y)

        # Blink detection (left eye)
        left_upper = landmarks[159]
        left_lower = landmarks[145]
        blink_dist = euclidean_dist(left_upper, left_lower)

        # Draw for debug
        cx, cy = int(right_iris.x * img_w), int(right_iris.y * img_h)
        cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
        cv2.putText(frame, f'Blink Dist: {blink_dist:.3f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

        # Threshold for blink detection
        if blink_dist < 0.025:
            current_time = time.time()
            if current_time - last_click_time > blink_click_cooldown:
                pyautogui.click()
                last_click_time = current_time
                cv2.putText(frame, 'CLICKED!', (cx, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow("Eye Controlled Mouse (Blink to Click)", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()

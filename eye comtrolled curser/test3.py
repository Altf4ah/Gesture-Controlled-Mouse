import cv2
import mediapipe as mp
import pyautogui
import time
import math

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1)

# Screen resolution
screen_w, screen_h = pyautogui.size()

# Webcam setup
cap = cv2.VideoCapture(0)

# Time management to prevent multiple clicks
last_click_time = 0
click_cooldown = 1  # seconds

# Euclidean distance helper
def euclidean_dist(p1, p2):
    return math.hypot(p1.x - p2.x, p1.y - p2.y)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    img_h, img_w, _ = frame.shape

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark

        # Iris center (right eye) - used to move cursor
        right_iris = landmarks[474]
        cursor_x = screen_w * right_iris.x
        cursor_y = screen_h * right_iris.y
        pyautogui.moveTo(cursor_x, cursor_y)

        # Landmark pairs for eye opening detection
        left_eye_upper = landmarks[159]
        left_eye_lower = landmarks[145]
        right_eye_upper = landmarks[386]
        right_eye_lower = landmarks[374]

        left_eye_dist = euclidean_dist(left_eye_upper, left_eye_lower)
        right_eye_dist = euclidean_dist(right_eye_upper, right_eye_lower)

        # Wink detection (right eye closed, left eye open)
        if right_eye_dist < 0.025 and left_eye_dist > 0.03:
            current_time = time.time()
            if current_time - last_click_time > click_cooldown:
                pyautogui.click()
                last_click_time = current_time
                cv2.putText(frame, "WINK CLICK", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Draw cursor point
        cx, cy = int(right_iris.x * img_w), int(right_iris.y * img_h)
        cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

    cv2.imshow("Eye Controlled Mouse with Wink Click", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()

import cv2
import mediapipe as mp
import pyautogui

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# Get screen resolution
screen_w, screen_h = pyautogui.size()

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    img_h, img_w, _ = frame.shape

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Landmark 474 is near right eye iris center
            # Landmark 145 is near left eye iris center
            # Use either depending on webcam orientation
            eye = face_landmarks.landmark[474]

            x = int(eye.x * img_w)
            y = int(eye.y * img_h)

            # Draw circle for eye tracking (debug)
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

            # Convert to screen coordinates
            screen_x = screen_w * eye.x
            screen_y = screen_h * eye.y
            pyautogui.moveTo(screen_x, screen_y)

    cv2.imshow("Eye Controlled Mouse", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()


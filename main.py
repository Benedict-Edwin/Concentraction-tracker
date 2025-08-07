import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

mp_drawing = mp.solutions.drawing_utils

# Define constants for eye aspect ratio (EAR) threshold
EYE_CLOSED_THRESHOLD = 0.25

# Get eye landmarks (left & right) from MediaPipe (based on official documentation)
LEFT_EYE_LANDMARKS = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_LANDMARKS = [362, 385, 387, 263, 373, 380]

def calculate_EAR(landmarks, eye_indices):
    """Compute Eye Aspect Ratio (EAR) to detect eye openness."""
    left = np.array([landmarks[eye_indices[0]].x, landmarks[eye_indices[0]].y])
    right = np.array([landmarks[eye_indices[3]].x, landmarks[eye_indices[3]].y])
    top = (np.array([landmarks[eye_indices[1]].x, landmarks[eye_indices[1]].y]) + 
           np.array([landmarks[eye_indices[2]].x, landmarks[eye_indices[2]].y])) / 2
    bottom = (np.array([landmarks[eye_indices[4]].x, landmarks[eye_indices[4]].y]) + 
              np.array([landmarks[eye_indices[5]].x, landmarks[eye_indices[5]].y])) / 2

    hor_dist = np.linalg.norm(left - right)
    ver_dist = np.linalg.norm(top - bottom)

    return ver_dist / hor_dist

def get_head_orientation(landmarks, img_w, img_h):
    """Estimate head tilt from nose and cheek points (basic)."""
    nose_tip = landmarks[1]
    left_cheek = landmarks[234]
    right_cheek = landmarks[454]

    nose_x = nose_tip.x * img_w
    cheek_dist = (right_cheek.x - left_cheek.x)

    if cheek_dist == 0:
        return 0
    rel_pos = (nose_tip.x - left_cheek.x) / cheek_dist
    if rel_pos < 0.3:
        return "Looking Right"
    elif rel_pos > 0.7:
        return "Looking Left"
    else:
        return "Looking Center"

# Initialize webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Flip frame for mirror view
    frame = cv2.flip(frame, 1)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_h, img_w, _ = frame.shape

    # Detect face mesh
    results = face_mesh.process(img_rgb)

    status = "No Face Detected"

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS)

            # Calculate eye openness
            EAR_left = calculate_EAR(face_landmarks.landmark, LEFT_EYE_LANDMARKS)
            EAR_right = calculate_EAR(face_landmarks.landmark, RIGHT_EYE_LANDMARKS)
            avg_EAR = (EAR_left + EAR_right) / 2

            # Head orientation
            head_dir = get_head_orientation(face_landmarks.landmark, img_w, img_h)

            # Apply concentration rules
            if avg_EAR < EYE_CLOSED_THRESHOLD:
                status = "Drowsy 😴"
            elif head_dir != "Looking Center":
                status = f"Distracted 🧐 ({head_dir})"
            else:
                status = "Focused ✅"

    # Display status
    cv2.putText(frame, f"Status: {status}", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if status == "Focused ✅" else (0, 0, 255), 2)
    cv2.imshow("Concentration Tracker", frame)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

mp_drawing = mp.solutions.drawing_utils

# Define constants for eye aspect ratio (EAR) threshold
EYE_CLOSED_THRESHOLD = 0.25

# Get eye landmarks (left & right) from MediaPipe (based on official documentation)
LEFT_EYE_LANDMARKS = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_LANDMARKS = [362, 385, 387, 263, 373, 380]

def calculate_EAR(landmarks, eye_indices):
    """Compute Eye Aspect Ratio (EAR) to detect eye openness."""
    left = np.array([landmarks[eye_indices[0]].x, landmarks[eye_indices[0]].y])
    right = np.array([landmarks[eye_indices[3]].x, landmarks[eye_indices[3]].y])
    top = (np.array([landmarks[eye_indices[1]].x, landmarks[eye_indices[1]].y]) + 
           np.array([landmarks[eye_indices[2]].x, landmarks[eye_indices[2]].y])) / 2
    bottom = (np.array([landmarks[eye_indices[4]].x, landmarks[eye_indices[4]].y]) + 
              np.array([landmarks[eye_indices[5]].x, landmarks[eye_indices[5]].y])) / 2

    hor_dist = np.linalg.norm(left - right)
    ver_dist = np.linalg.norm(top - bottom)

    return ver_dist / hor_dist

def get_head_orientation(landmarks, img_w, img_h):
    """Estimate head tilt from nose and cheek points (basic)."""
    nose_tip = landmarks[1]
    left_cheek = landmarks[234]
    right_cheek = landmarks[454]

    nose_x = nose_tip.x * img_w
    cheek_dist = (right_cheek.x - left_cheek.x)

    if cheek_dist == 0:
        return 0
    rel_pos = (nose_tip.x - left_cheek.x) / cheek_dist
    if rel_pos < 0.3:
        return "Looking Right"
    elif rel_pos > 0.7:
        return "Looking Left"
    else:
        return "Looking Center"

# Initialize webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Flip frame for mirror view
    frame = cv2.flip(frame, 1)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_h, img_w, _ = frame.shape

    # Detect face mesh
    results = face_mesh.process(img_rgb)

    status = "No Face Detected"

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS)

            # Calculate eye openness
            EAR_left = calculate_EAR(face_landmarks.landmark, LEFT_EYE_LANDMARKS)
            EAR_right = calculate_EAR(face_landmarks.landmark, RIGHT_EYE_LANDMARKS)
            avg_EAR = (EAR_left + EAR_right) / 2

            # Head orientation
            head_dir = get_head_orientation(face_landmarks.landmark, img_w, img_h)

            # Apply concentration rules
            if avg_EAR < EYE_CLOSED_THRESHOLD:
                status = "Drowsy 😴"
            elif head_dir != "Looking Center":
                status = f"Distracted 🧐 ({head_dir})"
            else:
                status = "Focused ✅"

    # Display status
    cv2.putText(frame, f"Status: {status}", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if status == "Focused ✅" else (0, 0, 255), 2)
    cv2.imshow("Concentration Tracker", frame)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()


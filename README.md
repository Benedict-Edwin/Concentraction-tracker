Concentration Tracker using Face Mesh
This application uses computer vision to track a user's concentration level by monitoring eye openness and head orientation in real-time.

Features
Eye Openness Detection: Uses Eye Aspect Ratio (EAR) to detect if eyes are closed (drowsiness)

Head Orientation Tracking: Determines if the user is looking left, right, or center

Concentration Status: Provides real-time feedback on focus level:

✅ Focused (eyes open and looking forward)

😴 Drowsy (eyes closed)

� Distracted (head turned left or right)

Requirements
Python 3.6+

OpenCV (pip install opencv-python)

MediaPipe (pip install mediapipe)

NumPy (pip install numpy)

How It Works
Captures video from your webcam

Uses MediaPipe's Face Mesh to detect facial landmarks

Calculates:

Eye Aspect Ratio (EAR) for both eyes

Head orientation based on nose and cheek positions

Determines concentration status based on thresholds

Customization
Adjust EYE_CLOSED_THRESHOLD (default: 0.25) for sensitivity to eye closure

Modify landmark indices in LEFT_EYE_LANDMARKS and RIGHT_EYE_LANDMARKS if needed

Usage
Run the script: python concentration_tracker.py

Face the camera directly for best results

The system will display your concentration status in real-time

Press ESC to exit

Potential Applications
Study/work focus monitoring

Driver drowsiness detection

Attention tracking during online meetings

Note: For production use, consider adding temporal smoothing and more sophisticated head pose estimation.

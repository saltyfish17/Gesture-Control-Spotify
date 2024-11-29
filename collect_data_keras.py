import cv2
import numpy as np
import os
import mediapipe as mp
from tensorflow.keras.models import load_model

# Load your trained model
model = load_model('best_gesture_model.keras')  # Replace with your model path

# Mediapipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Gesture labels
GESTURES = ["thumbs_up", "peace", "stop", "fist", "pinch", "left", "right", "ok", "none"]

# Directory for saving data
DATA_DIR = "gesture_data"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Normalize landmarks function (unchanged)
def normalize_to_wrist(landmarks):
    """
    Normalize landmarks relative to the wrist position.
    """
    wrist = landmarks[0]  # Wrist is the first landmark
    normalized_landmarks = landmarks - wrist  # Center landmarks to wrist
    return normalized_landmarks.flatten()

# Save landmarks to CSV
def save_landmarks(landmarks, gesture_name):
    gesture_dir = os.path.join(DATA_DIR, gesture_name)
    if not os.path.exists(gesture_dir):
        os.makedirs(gesture_dir)
    file_name = os.path.join(gesture_dir, f"{len(os.listdir(gesture_dir))}.csv")
    np.savetxt(file_name, landmarks, delimiter=',')

# Real-time data collection
def collect_data():
    cap = cv2.VideoCapture(0)  # Open webcam
    current_label = None
    print(f"Starting data collection. Press a number key to select a gesture: {GESTURES}")
    
    with mp_hands.Hands(model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Ignoring empty frame.")
                continue
            
            # Flip and process the frame
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Extract and normalize landmarks
                    landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
                    normalized_landmarks = normalize_to_wrist(landmarks)

                    # Save landmarks if a label is selected
                    if current_label is not None:
                        save_landmarks(normalized_landmarks, GESTURES[current_label])
                        cv2.putText(frame, f"Recording Gesture: {GESTURES[current_label]}", 
                                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    # Draw landmarks on the frame
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Display the frame
            cv2.imshow('Gesture Data Collection', frame)
            
            # Key bindings
            key = cv2.waitKey(5) & 0xFF
            if key == ord('q'):  # Quit
                break
            elif key == ord('c'):  # Clear current label
                current_label = None
                print("Gesture label cleared.")
            elif key in map(lambda i: ord(str(i)), range(len(GESTURES))):  # Select gesture
                current_label = int(chr(key))
                print(f"Recording gesture: {GESTURES[current_label]}")
    
    cap.release()
    cv2.destroyAllWindows()

collect_data()

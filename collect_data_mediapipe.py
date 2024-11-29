import cv2
import mediapipe as mp
import numpy as np
import os
import csv

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Gesture labels and directory setup
GESTURES = ["thumbs_up", "peace", "stop", "fist", "pinch", "left", "right", "inver_ok", "L", "none"]  # Define your gestures here
DATA_DIR = "gesture_data"

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Helper function to save landmarks for a specific gesture
def save_landmarks(landmarks, gesture_name):
    gesture_dir = os.path.join(DATA_DIR, gesture_name)
    if not os.path.exists(gesture_dir):
        os.makedirs(gesture_dir)
    file_name = os.path.join(gesture_dir, f"{len(os.listdir(gesture_dir))}.csv")
    np.savetxt(file_name, landmarks, delimiter=',')

# Data collection function
def collect_data():
    cap = cv2.VideoCapture(0)  # Open webcam
    current_label = None
    print(f"Starting data collection. Press a number key to select a gesture: {GESTURES}")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Flip and process the frame
        frame = cv2.flip(frame, 1)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks on the frame
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Extract and normalize landmarks
                landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
                
                # Save landmarks for the current gesture
                if current_label is not None:
                    save_landmarks(landmarks, GESTURES[current_label])
                    cv2.putText(frame, f"Recording Gesture: {GESTURES[current_label]}", 
                                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Display the frame
        cv2.imshow("Hand Gesture Data Collection", frame)
        
        # Key bindings for control
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # Quit
            break
        elif key == ord('c'):  # Clear label
            current_label = None
            print("Gesture label cleared.")
        elif key in map(lambda i: ord(str(i)), range(len(GESTURES))):  # Select gesture
            current_label = int(chr(key))
            print(f"Recording gesture: {GESTURES[current_label]}")

    cap.release()
    cv2.destroyAllWindows()

collect_data()

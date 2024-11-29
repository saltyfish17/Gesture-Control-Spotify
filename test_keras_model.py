import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

# To exit, press `Esc`

# Load the trained model
model = load_model('best_gesture_model.keras')  # Replace with your model path

# Mediapipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def normalize_to_wrist(landmarks):
    """
    Normalize landmarks relative to the wrist position.
    """
    wrist = landmarks[0]  # Wrist is the first landmark
    normalized_landmarks = landmarks - wrist  # Center landmarks to wrist
    return normalized_landmarks.flatten()

# Real-time gesture recognition
import cv2

GESTURES = ["thumbs_up", "peace", "stop", "fist", "pinch", "left", "right", "ok", "inverted_ok", "L", "four", "six", "none"]  # Define your gestures here

cap = cv2.VideoCapture(0)  # Open webcam
with mp_hands.Hands(model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty frame.")
            continue

        # Flip and convert the frame
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame for hands
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Extract landmark coordinates
                landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])

                # Normalize landmarks relative to the wrist
                landmarks_normalized = normalize_to_wrist(landmarks)

                # Prepare input for the model
                input_data = landmarks_normalized.reshape(1, -1)  # Shape: (1, 63)

                # Predict the gesture
                predictions = model.predict(input_data)
                predicted_gesture = GESTURES[np.argmax(predictions)]
                confidence = np.max(predictions)

                # Display the predicted gesture
                cv2.putText(frame, f"{predicted_gesture} ({confidence:.2f})",
                            (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                # Draw landmarks
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Show the frame
        cv2.imshow('Hand Gesture Recognition', frame)

        if cv2.waitKey(5) & 0xFF == 27:  # Press 'Esc' to exit
            break

cap.release()
cv2.destroyAllWindows()

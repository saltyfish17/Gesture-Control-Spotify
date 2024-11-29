import cv2
import os
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import pyautogui
import psutil
import subprocess
import time
import tkinter as tk
from tkinter import ttk
import json
import pygetwindow as gw
import win32gui
import win32con
import win32api
from PIL import Image, ImageTk

# ========================
# Load Model and Setup
# ========================
model = load_model('best_gesture_model.keras')

# Mediapipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

GESTURES = ['thumbs_up', 'peace', 'stop', 'fist', 'pinch', 'left', 'right', 'ok', 'inverted_ok', 'L', 'four', 'six', 'none']
gesture_actions = {
    'thumbs_up': 'spotify_toggle',
    'peace': 'volume_up',
    'stop': 'play_pause',
    'fist': 'close_app',
    'right': 'repeat',
    'ok': 'shuffle',
    'L': 'volume_down',
    'four': 'prev_track',
    'six': 'next_track'
}

# ========================
# Save/Load Configuration
# ========================
def save_gesture_config():
    with open('gesture_config.json', 'w') as f:
        json.dump(gesture_actions, f)

def load_gesture_config():
    global gesture_actions
    try:
        with open('gesture_config.json', 'r') as f:
            gesture_actions = json.load(f)
    except FileNotFoundError:
        print("No configuration file found. Using default mappings.")

load_gesture_config()

# ========================
# Perform Action
# ========================

# Store the last time a gesture was executed to handle cooldowns
last_action_time = {'general': 0, 'volume': 0, 'media': 0}
cooldown_map = {'general': 1.0, 'media': 1.0, 'volume': 0.1}  # Cooldowns for various actions

def perform_action(gesture):
    global last_action_time

    gesture = gesture_actions.get(gesture, None)
    if not gesture:
        return

    # Get current time
    current_time = time.time()

    # Media Control (Play/Pause, Next/Prev Track, etc.)
    if current_time - last_action_time['media'] > cooldown_map['media']:
        if gesture == 'play_pause':
            print("Toggling Play/Pause.")
            pyautogui.press('space')  # Spacebar to play/pause media
        elif gesture == 'next_track':
            print("Skipping to next track.")
            pyautogui.hotkey('ctrl', 'right')  # Ctrl + Right Arrow for next track
        elif gesture == 'prev_track':
            print("Going to previous track.")
            pyautogui.hotkey('ctrl', 'left')  # Ctrl + Left Arrow for previous track
        elif gesture == 'shuffle':
            print("Toggling shuffle.")
            pyautogui.hotkey('ctrl', 's')  # Ctrl + S for shuffle toggle
        elif gesture == 'repeat':
            print("Toggling repeat.")
            pyautogui.hotkey('ctrl', 'r')  # Ctrl + R for repeat toggle
        
        # Update last action time to prevent rapid-fire actions
        last_action_time['media'] = current_time
        return

    # Open or quit Spotify (toggle action)
    if current_time - last_action_time['general'] > cooldown_map['general']:
        if gesture == 'spotify_toggle':
            print("Toggling Spotify.")
            if is_spotify_running():
                close_spotify()
            else:
                open_spotify()
        elif gesture == 'close_app':
            print("Closing app.")
            pyautogui.hotkey('esc')  # Ctrl + R for repeat toggle

        # Update last action time for general actions
        last_action_time['general'] = current_time
        return

    if current_time - last_action_time['volume'] > cooldown_map['volume']:
            # Handle brightness adjustments
            if gesture == 'volume_up':
                print("Increasing system volume.")
                adjust_volume(increase=True)
            elif gesture == 'volume_down':
                print("Decreasing system volume.")
                adjust_volume(increase=False)

            # Update last action time for general actions
            last_action_time['volume'] = current_time

    return

def get_spotify_path():
    """Locate Spotify executable dynamically."""
    # Check default installation paths
    possible_paths = [
        os.path.join(os.getenv('APPDATA'), "Spotify", "Spotify.exe"),
        os.path.join(os.getenv('LOCALAPPDATA'), "Spotify", "Spotify.exe"),
    ]
    for path in possible_paths:
        if os.path.exists(path):
            return path
    raise FileNotFoundError("Spotify executable not found in default locations.")

def open_spotify():
    """Open Spotify if not already running."""
    if not is_spotify_running():
        try:
            spotify_path = get_spotify_path()
            subprocess.Popen([spotify_path])
            print("Opening Spotify.")
        except FileNotFoundError as e:
            print(e)
    else:
        # If Spotify is running but minimized, restore the window
        restore_spotify_window()

def close_spotify():
    """Close Spotify if it's running."""
    for proc in psutil.process_iter(['pid', 'name']):
        if 'spotify' in proc.info['name'].lower():
            proc.terminate()
    print("Spotify closed.")

def toggle_spotify():
    """Toggle Spotify (open if closed, close if open)."""
    if is_spotify_running():
        close_spotify()
    else:
        open_spotify()

def is_spotify_running():
    """Check if Spotify is running."""
    for proc in psutil.process_iter(['pid', 'name']):
        if 'spotify' in proc.info['name'].lower():
            return True
    return False

def restore_spotify_window():
    """Restore Spotify window if it's minimized."""
    try:
        spotify_window = gw.getWindowsWithTitle('Spotify')[0]  # Get the Spotify window
        if spotify_window.isMinimized:  # Check if it is minimized
            spotify_window.restore()  # Restore the window
            print("Restoring minimized Spotify window.")
        spotify_window.activate()  # Bring it to the front
    except IndexError:
        print("Spotify window not found.")


def adjust_volume(increase=True):
    """Adjust screen brightness (Windows specific)"""
    # Windows-specific approach using 'pyautogui' to simulate keys
    if increase:
        pyautogui.press('volumeup')  
    else:
        pyautogui.press('volumedown')  
        
# ========================
# Gesture Configuration GUI
# ========================

# Save function placeholder
def save_mapping_and_close(root):
    for gesture, var in gesture_vars.items():
        gesture_actions[gesture] = var.get()
    save_gesture_config()
    print("Updated Gesture Actions:", gesture_actions)
    root.destroy()

# Function to create the tutorial page
def create_tutorial(root):
    tutorial_frame = tk.Frame(root, padx=20, pady=20)
    tutorial_frame.grid(row=0, column=0, sticky="nsew")

    # Create a scrollable frame
    canvas = tk.Canvas(tutorial_frame)
    scrollbar = ttk.Scrollbar(tutorial_frame, orient="vertical", command=canvas.yview)
    scrollable_frame = tk.Frame(canvas)

    scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))  

    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")  

    canvas.configure(yscrollcommand=scrollbar.set)

    canvas.grid(row=0, column=0, sticky="nsew")
    scrollbar.grid(row=0, column=1, sticky="ns")  

    # Ensure that the tutorial frame resizes properly
    tutorial_frame.grid_rowconfigure(0, weight=1)
    tutorial_frame.grid_columnconfigure(0, weight=1)

    # Title
    tk.Label(scrollable_frame, text="Gesture Control Setup - Tutorial", font=("Helvetica", 16, "bold")).grid(row=0, column=0, columnspan=2, pady=10)

    # Instructions
    instructions = [
        ("Step 1", "Select a gesture you want to configure."),
        ("Step 2", "Choose the action you want to assign to that gesture."),
        ("Step 3", "Click 'Save and Start' to save your configuration and begin.")
    ]
    
    # Display instructions in grid
    row = 1
    for step, description in instructions:
        tk.Label(scrollable_frame, text=step, font=("Helvetica", 12, "bold")).grid(row=row, column=0, sticky="w", pady=5)
        tk.Label(scrollable_frame, text=description, font=("Helvetica", 12)).grid(row=row, column=1, sticky="w", pady=5)
        row += 1

    row += 1

    actions = [
        ("Spotify Toggle (spotify_toggle)", "Open/Close Spotify"),
        ("Play/Pause (play_pause)", "Play or Pause the Curent Track"),
        ("Volume Up/Down (volume_up/down)", "Increase or Decrease the Volume"),
        ("Volume Up/Down (volume_up/down)", "Increase or Decrease the Volume"),
        ("Next/Previous Track (next/prev_track)", "Skip to the Next or Previous Track"),
        ("Repeat (repeat)", "Spotify will Repeat Playlist or Current Track"),
        ("Close App (close_app)", "Close Gesture Detection")
    ]
    
    # Display instructions in grid
    for step, description in actions:
        tk.Label(scrollable_frame, text=step, font=("Helvetica", 10, "bold")).grid(row=row, column=0, sticky="w", pady=5)
        tk.Label(scrollable_frame, text=description, font=("Helvetica", 10)).grid(row=row, column=1, sticky="w", pady=5)
        row += 1

    row += 1

    # List all jpg image files in the 'images' folder
    image_files = [f for f in os.listdir("images") if f.endswith('.jpg')]

    # Display images and their captions
    image_row = row  # Start the images after the instructions
    col = 0
    max_columns = 3  # Number of images per row
    for image_file in image_files:
        try:
            # Load and resize the image
            image = Image.open(f"images/{image_file}")
            image = image.resize((200, 200))  # Resize image to 200x200
            photo = ImageTk.PhotoImage(image)  # Convert the image to a format Tkinter can use

            # Create a label for the image
            image_label = tk.Label(scrollable_frame, image=photo)
            image_label.image = photo  # Keep a reference to prevent garbage collection
            image_label.grid(row=image_row, column=col, padx=10, pady=10)

            # Create a label for the image filename as caption
            caption = tk.Label(scrollable_frame, text=image_file, font=("Helvetica", 10), anchor="center")
            caption.grid(row=image_row + 1, column=col, padx=10, pady=5)

            # Move to next column and row if necessary
            col += 1
            if col >= max_columns:  # Start a new row
                col = 0
                image_row += 2  # Move down two rows (one for the image, one for the caption)
        except Exception as e:
            print(f"Error loading image {image_file}: {e}")

    # Back button
    def go_back():
        tutorial_frame.grid_forget()
        create_gui(root)  # Ensure create_gui is defined elsewhere

    back_button = tk.Button(scrollable_frame, text="Back to Configuration", command=go_back, bg="#4CAF50", fg="white")
    back_button.grid(row=image_row, column=0, columnspan=3, pady=10)


# Function to create the main configuration page
def create_gui(root=None):
    if root is None:
        root = tk.Tk()
        root.title("Gesture Control Configuration")

        # Set the window to half screen and center it
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        window_width = int(screen_width / 1.5)
        window_height = int(screen_height / 1.3)
        position_top = int(screen_height / 30)
        position_left = int(screen_width / 4)
        root.geometry(f'{window_width}x{window_height}+{position_left}+{position_top}')
        
        root.config(bg="#f0f0f0")

    # Add a main frame to center the widgets
    main_frame = tk.Frame(root, padx=20, pady=20, bg="#f0f0f0")
    main_frame.grid(row=0, column=0, sticky="nsew")

    # Title and tutorial button
    tk.Label(main_frame, text="Control Spotify with Hand Gestures", font=("Helvetica", 16, "bold")).grid(row=0, column=0, columnspan=2, pady=10)
    
    row = 1
    for gesture in GESTURES:
        tk.Label(main_frame, text=gesture, font=("Helvetica", 12)).grid(row=row, column=0, padx=10, pady=5)
        gesture_vars[gesture] = tk.StringVar(value=gesture_actions.get(gesture, 'None'))
        action_menu = ttk.Combobox(main_frame, textvariable=gesture_vars[gesture], values=ACTIONS, width=20)
        action_menu.grid(row=row, column=1, padx=10, pady=5)
        row += 1

    # Save and Close button
    save_button = tk.Button(main_frame, text="Save and Start", command=lambda: save_mapping_and_close(root), bg="#4CAF50", fg="white")
    save_button.grid(row=row, column=0, columnspan=2, pady=10)

    # Tutorial button below the configuration
    tutorial_button = tk.Button(main_frame, text="Tutorial", command=lambda: create_tutorial(root), bg="#FF5733", fg="white", font=("Helvetica", 14, "bold"))
    tutorial_button.grid(row=row + 1, column=0, columnspan=2, pady=20)
    
    # Configure grid weights for better resizing behavior
    root.grid_rowconfigure(0, weight=1)
    root.grid_columnconfigure(0, weight=1)
    
    root.mainloop()

gesture_vars = {}
ACTIONS = ['spotify_toggle', 'shuffle', 'play_pause', 'volume_up', 'volume_down', 'prev_track', 'next_track', 'repeat', 'close_app', 'None']

# Uncomment the next line to launch GUI
create_gui()

# ========================
# Real-time Gesture Recognition
# ========================

def normalize_to_wrist(landmarks):
    wrist = landmarks[0]
    normalized_landmarks = landmarks - wrist
    return normalized_landmarks.flatten()

def set_window_on_top(window_name):
    """
    Set the OpenCV window to always stay on top and position it at the bottom-right corner of the screen.
    """
    try:
        hwnd = win32gui.FindWindow(None, window_name)  # Get the handle of the OpenCV window
        if hwnd:
            # Get screen dimensions
            screen_width = win32api.GetSystemMetrics(win32con.SM_CXSCREEN)
            screen_height = win32api.GetSystemMetrics(win32con.SM_CYSCREEN)

            # Define window dimensions
            window_width = 480  # Adjust as needed
            window_height = 360  # Adjust as needed

            # Calculate bottom-right corner position
            x_pos = screen_width - window_width - 10
            y_pos = screen_height - window_height - 10

            # Set window position
            win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, x_pos, y_pos, window_width, window_height, 0)
    except Exception as e:
        print("Error setting window position:", e)

def focus_opencv_window(window_name):
    """Bring the OpenCV window to the foreground and make it active."""
    try:
        hwnd = win32gui.FindWindow(None, window_name)  # Get the handle of the OpenCV window
        if hwnd:
            win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)  # Restore the window if minimized
            win32gui.SetForegroundWindow(hwnd)  # Bring it to the foreground
    except Exception as e:
        print(f"Failed to focus the window '{window_name}':", e)

cap = cv2.VideoCapture(0)
window_name = 'Gesture Control'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

with mp_hands.Hands(model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty frame.")
            continue

        # Check if Spotify is running
        spotify_running = is_spotify_running()

        if not spotify_running:
            focus_opencv_window(window_name)  # Refocus OpenCV window if Spotify is closed

        # Flip the camera feed to the default front-facing view
        frame = cv2.flip(frame, 1)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
                landmarks_normalized = normalize_to_wrist(landmarks)
                input_data = landmarks_normalized.reshape(1, -1)

                predictions = model.predict(input_data)
                predicted_gesture = GESTURES[np.argmax(predictions)]
                confidence = np.max(predictions)

                if confidence > 0.8:
                    perform_action(predicted_gesture)

                # Display the predicted gesture
                cv2.putText(frame, f"{predicted_gesture} ({confidence:.2f})",
                            (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.imshow(window_name, frame)
        set_window_on_top(window_name)  # Keep the window at the bottom-right corner

        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()

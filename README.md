# Gesture-Control-Spotify

## Overview
This application enables users to control Spotify using hand gestures via a camera. The app maps predefined actions (e.g., Play, Pause, Next Track) to specific hand gestures, providing a hands-free way to interact with Spotify.

---

## Features
- **Control Spotify Actions**:
    - Open/Close Spotify
    - Play/Pause
    - Volume Control
    - Skip Tracks
    - Enable Repeat
    - Enable Shuffle
    - Close the application
- **Customizable Gesture Mapping**: Assign actions to gestures using the dropdown menu in the app interface.
- **Expandable**:
    - Add more actions in the `perform_action` function.
    - Add new gestures (aligned with model training).

---

## Prerequisites
1. Ensure Spotify is installed on your system.
2. Install Python and required libraries
   
---

## Usage
1. **Clone Repository**:
   ```bash
   git clone https://github.com/saltyfish17/Gesture-Control-Spotify.git
   cd Gesture-Control-Spotify
   ```
2. **Launch the App**:
   ```bash
   python app.py
   ```
3. **Map Actions to Gestures**:
   - Use the dropdown menu in the app interface to assign desired actions to gestures.
4. **Perform Gestures**:
   - Ensure your hand is centered and fully visible to the camera for best performance.

### Supported Actions
| Action                      | Gesture Description                                |
|-----------------------------|----------------------------------------------------|
| Spotify Toggle (`spotify_toggle`) | Open or Close Spotify                          |
| Play/Pause (`play_pause`)         | Play or Pause the Current Track               |
| Volume Up/Down (`volume_up/down`) | Increase or Decrease the Volume               |
| Next/Previous Track (`next/prev_track`) | Skip to the Next or Previous Track        |
| Repeat (`repeat`)                  | Repeat Playlist or Current Track             |
| Shuffle (`shuffle`)                  | Set Playlist to Shuffle mode            |
| Close App (`close_app`)            | Stop Gesture Detection and Exit the Application |

---

## Gestures
| Gesture Name      | Description |
|-------------------|-----------------|
| `thumbs_up`       | 👍             |
| `peace`           | ✌️            |
| `stop`            | 🖐️            |
| `fist`            | ✊(Palm hidden from camera)             |
| `pinch`           | Fingers curled over thumb, palm visible to camera             |
| `left`            | 👉             |
| `right`           | 👈             |
| `ok`              | 👌             |
| `inverted_ok`     | 👌 but back of hand facing the camera            |
| `L`               | Thumb and index outstretched             |
| `four`            | Thumb folded towards palm, four fingers out             |
| `six`             | 🤟             |
| `none`            | No Gesture     |

> **Note**: Gesture images are available in the `images` folder and the app tutorial page.

---

## Model Training
### Data Collection
1. **Using `mediapipe`**:
   - Run `collect_data_mediapipe.py`.
   - Specify gestures in the script and use keyboard input to select the list index for each gesture during collection.
   - The list can contain a maximum of 10 gestures, but to record more, remove old items.
   - Press `c` to clear selection and `q` to quit.
2. **Using a Trained Model**:
   - If the trained model mistakenly detects facial features, run `collect_data_keras.py` to include a separate class (`none`) for facial detection.

### Model Training
- Train the gesture model using `train_keras_colab.ipynb` on Google Colab or a local machine with GPU support.
- Save the trained model in the same directory as `app.py` and `test_keras_model.py`.

### Testing the Model
- To test the model independently, use the `test_keras_model.py` script:
   ```bash
   python test_keras_model.py
   ```
- Press `Esc` to quit.

---

## Notes
- **Camera Setup**: Ensure the hand is centered and fully visible for accurate detection.
- **Expandable**:
   - More gestures can be added to the `gestures` list (aligned with the number of classes the model was trained on).
   - Define additional actions in the `perform_action` function.

---

Enjoy controlling Spotify with your gestures! 🎶

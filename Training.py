import cv2
import numpy as np
import os
import mediapipe as mp
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
import time  # Add this import statement at the top

# Set up folders and paths
current_word_index = 0
DATA_PATH = 'MP_Data'
actions = np.array(['HELLO', 'I LOVE YOU', 'PLEASE', 'GOOD MORNING', 'FINISH', 'HELP', 'OK'])
no_sequences = 50
sequence_length = 30

# Create necessary folders for data collection
for action in actions:
    for sequence in range(no_sequences):
        folder_path = os.path.join(DATA_PATH, action, str(sequence))
        os.makedirs(folder_path, exist_ok=True)


# Check if training data needs collection for a specific action and sequence
def needs_data_collection(action, sequence):
    return not all(
        os.path.isfile(os.path.join(DATA_PATH, action, str(sequence), f"{frame_num}.npy"))
        for frame_num in range(sequence_length)
    )


# Data collection loop
def collect_data(cap, holistic):
    global current_word_index
    action = actions[current_word_index]
    collection_complete = False  # Flag to indicate data collection completion
    for sequence in range(no_sequences):
        print(f"Collecting data for {action} - Sequence {sequence}")
        for frame_num in range(sequence_length):
            if needs_data_collection(action, sequence, frame_num):
                ret, frame = cap.read()
                image, results = mediapipe_detection(frame, holistic)
                draw_styled_landmarks(image, results)  # Draw landmarks on the frame
                cv2.imshow('Frame', image)  # Show the frame with landmarks
                save_keypoints(results, action, sequence, frame_num)

    print(f"Data collection for {action} completed. Press 'N' to go to the next word, 'Q' to quit, and 'A' to Train")
    key = cv2.waitKey(0)
    if key == ord('q') or key == 27:
        return
    elif key == ord('n') or key == ord('N'):
        current_word_index = (current_word_index + 1)
        print(f"Current word index: {current_word_index} ({actions[current_word_index]})")
        start_collection_on_key_press(cap, holistic)
    elif key == ord('a') or key == ord('A'):
        cap.release()
        cv2.destroyAllWindows()
        # Load and preprocess data
        X_train, X_test, y_train, y_test = load_data()
        # Build and train LSTM model
        build_train_model(X_train, X_test, y_train, y_test)


def needs_data_collection(action, sequence, frame_num):
    file_path = os.path.join(DATA_PATH, action, str(sequence), f"{frame_num}.npy")
    return not os.path.isfile(file_path)


# Function to save keypoints as .npy files
def save_keypoints(results, action, sequence, frame_num):
    keypoints = extract_keypoints(results)
    npy_path = os.path.join(DATA_PATH, action, str(sequence), f"{frame_num}.npy")
    np.save(npy_path, keypoints)


# Load and preprocess data for training
def load_data():
    sequences, labels = [], []
    label_map = {label: num for num, label in enumerate(actions)}

    for action in actions:
        for sequence in range(no_sequences):
            window = []
            for frame_num in range(sequence_length):
                res = np.load(os.path.join(DATA_PATH, action, str(sequence), f"{frame_num}.npy"))
                window.append(res)
            sequences.append(window)
            labels.append(label_map[action])

    X = np.array(sequences)
    y = to_categorical(labels, num_classes=len(actions))

    return train_test_split(X, y, test_size=0.05)


# Build and train LSTM model
def build_train_model(X_train, X_test, y_train, y_test):
    log_dir = os.path.join('Logs')
    tb_callback = TensorBoard(log_dir=log_dir)

    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(sequence_length, 1662)),
        LSTM(128, return_sequences=True),
        LSTM(64),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(len(actions), activation='softmax')
    ])

    model.compile(optimizer="Adam", loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=2000, callbacks=[tb_callback], validation_data=(X_test, y_test))
    model.save('action_recognition_model.keras')
    return model


# Helper functions for Mediapipe processing
def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in
                     results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 4)
    face = np.array([[res.x, res.y, res.z] for res in
                     results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468 * 3)
    lh = np.array([[res.x, res.y, res.z] for res in
                   results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
    rh = np.array([[res.x, res.y, res.z] for res in
                   results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(
        21 * 3)
    return np.concatenate([pose, face, lh, rh])


def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False  # Image is no longer writeable
    results = model.process(image)  # Make prediction
    image.flags.writeable = True  # Image is now writeable
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # COLOR COVERSION RGB 2 BGR
    return image, results


def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION)  # Draw face connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)  # Draw pose connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks,
                              mp_holistic.HAND_CONNECTIONS)  # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks,
                              mp_holistic.HAND_CONNECTIONS)  # Draw right hand connections


def draw_styled_landmarks(image, results):
    # Draw face connections
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
                              mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                              mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1)
                              )
    # Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
                              )
    # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
                              )
    # Draw right hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                              )


def start_collection_on_key_press(cap, holistic):
    print("Press 'S' to start data collection.")
    while True:
        ret, frame = cap.read()
        cv2.imshow('Frame', frame)
        key = cv2.waitKey(1)
        if key == ord('s') or key == ord('S'):
            collect_data(cap, holistic)
            break
        elif key == ord('q') or key == 27:
            break
    print("Press 'A' to train the model.")


if __name__ == "__main__":
    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    start_collection_on_key_press(cap, holistic)

    cap.release()
    cv2.destroyAllWindows()

import uuid

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk

from PIL import Image, ImageTk
import mediapipe as mp
from tensorflow.keras.models import load_model
from googletrans import Translator
from gtts import gTTS
import time
import os
from language_api import LanguageAPI  # Import LanguageAPI from language_api.py
import pygame

# Load saved model
model = load_model('action_recognition_model.h5')

# Load Mediapipe and define necessary functions
mp_holistic = mp.solutions.holistic

def cleanup_audio_files(directory):
    try:
        # Get a list of all files in the directory
        file_list = os.listdir(directory)

        # Iterate through the files and remove audio files
        for file_name in file_list:
            if file_name.endswith(".mp3"):
                file_path = os.path.join(directory, file_name)
                os.remove(file_path)
                print(f"Removed: {file_path}")
    except Exception as e:
        print(f"Error cleaning up audio files: {e}")


def text_to_speech(text, lang):
    # Get the directory of the script
    temp_audio_dir = "Data_Audio"
    os.makedirs(temp_audio_dir, exist_ok=True)  # Ensure the directory exists

    # Generate a unique filename for the temporary audio file
    audio_file_name = f"temp_audio_{uuid.uuid4().hex}.mp3"
    audio_file_path = os.path.join(temp_audio_dir, audio_file_name)

    # Initialize gTTS and save speech directly to the unique file
    tts = gTTS(text=text, lang=lang, slow=False)
    tts.save(audio_file_path)

    # Play the speech using pygame mixer
    pygame.mixer.init()
    pygame.mixer.music.load(audio_file_path)
    pygame.mixer.music.play()

    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)

def mediapipe_detection(image, holistic):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = holistic.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

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

# Real-time gesture recognition loop
def gesture_recognition(cap, holistic, model):
    # Initialize LanguageAPI
    language_api = LanguageAPI()

    sequence = []
    recognized_word = ""  # Variable to hold recognized word
    fps_start_time = time.time()
    fps_counter = 0
    translator = Translator()

    def update_frame(language_combobox):
        nonlocal recognized_word, fps_start_time, fps_counter
        ret, frame = cap.read()
        fps_counter += 1

        if fps_counter % 3 == 0:  # Process every third frame
            image, results = mediapipe_detection(frame, holistic)
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequences = sequence[-30:]

            if len(sequence) == 30:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                action_index = np.argmax(res)

                if res[action_index] > 0.5:
                    action = actions[action_index]
                    recognized_word = action

            # Check if recognized_word is not empty or None
            if recognized_word:
                dest_language = language_combobox.get()
                translated_word = translator.translate(recognized_word, dest=dest_language).text
                subtitle_label.config(text=f"Recognized Word: {translated_word}")

                # Update the position of ComboBox
                language_combobox.grid(row=1, column=1, padx=(20, 0))  # Adjust padx for spacing
            else:
                subtitle_label.config(text="Recognized Word: ")  # Reset the text if recognized_word is empty
                language_combobox.grid_forget()  # Hide ComboBox when no word recognized

            # Update the video frame
            frame = cv2.resize(frame, (640, 480))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)
            video_label.imgtk = imgtk
            video_label.config(image=imgtk)

        # Calculate and display FPS
        fps_end_time = time.time()
        fps = int(1 / (fps_end_time - fps_start_time))
        fps_label.config(text=f"FPS: {fps}")
        fps_start_time = fps_end_time

        video_label.after(10, update_frame, language_combobox)  # Call update_frame after 10ms

    # Create the GUI window
    root = tk.Tk()
    root.title("Real-time Gesture Recognition")

    # Create a label to display the video
    video_label = tk.Label(root)
    video_label.grid(row=0, column=0, columnspan=2)

    # Create a label to display recognized word
    subtitle_label = tk.Label(root, text="Recognized Word: ", font=("Arial", 18))
    subtitle_label.grid(row=1, column=0, padx=(10, 5))  # Adjust padx for spacing

    # Create a label to display FPS
    fps_label = tk.Label(root, text="FPS: ")
    fps_label.grid(row=2, column=0)

    # Initialize language ComboBox
    language_combobox = ttk.Combobox(root, values=language_api.get_language_names(), state="readonly",
                                     font=("Arial", 18))
    language_combobox.set('English')  # Default language selection
    language_combobox.grid(row=1, column=1, padx=(5, 0))  # Adjust padx for spacing

    # Start the gesture recognition loop
    update_frame(language_combobox)

    def tts_button_callback():
        recognized_word = subtitle_label.cget("text").split(": ")[1]  # Get recognized word from label
        lang_code = language_api.get_language_code_by_name(language_combobox.get())
        text_to_speech(recognized_word, lang=lang_code)

    # Create a button for TTS
    tts_button = tk.Button(root, text="Text to Speech", command=tts_button_callback)
    tts_button.grid(row=2, column=1, pady=(10, 0))  # Adjust pady for spacing
    root.mainloop()


# Main testing program
if __name__ == "__main__":
    cleanup_audio_files("Data_Audio")
    cap = cv2.VideoCapture(0)
    holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    actions = np.array(['HELLO', 'THANK YOU', 'PLEASE'])

    gesture_recognition(cap, holistic, model)

    cap.release()
    cv2.destroyAllWindows()

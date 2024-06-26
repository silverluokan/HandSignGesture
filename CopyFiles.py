import shutil

def copy_folder_contents(source_folder, destination_folder):
    # Copy the contents of the source folder to the destination folder
    try:
        shutil.copytree(source_folder, destination_folder)
        print("Folder contents copied successfully!")
    except OSError as e:
        print(f"Error: {e}")


for i in range(0, 20):
    source_folder = f'MP_Data_Words/THANK YOU/{i}'
    destination_folder = f'MP_Data_Words/THANK YOU//{i+40}'

    copy_folder_contents(source_folder, destination_folder)

# Example usage

#TRAININGWORDS

# import cv2
# import numpy as np
# import os
# import mediapipe as mp
#
# from PIL import Image, ImageTk
# from sklearn.model_selection import train_test_split
# from tensorflow.keras.utils import to_categorical
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense
# from tensorflow.keras.callbacks import TensorBoard
# import customtkinter as ctk
#
# # Set up folders and paths
# DATA_PATH = 'MP_Data_Words'
# actions = np.array([action for action in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, action))])
# no_sequences = 120
# sequence_length = 30
# current_word_index = 0  # Initialize current_word_index globally
#
# # Create necessary folders for data collection
# for action in actions:
#     for sequence in range(no_sequences):
#         folder_path = os.path.join(DATA_PATH, action, str(sequence))
#         os.makedirs(folder_path, exist_ok=True)
#
# def needs_data_collection(action, sequence, frame_num):
#     """
#     Always return True to overwrite .npy files.
#     """
#     return True
#
# def save_keypoints(results, action, sequence, frame_num):
#     """
#     Extract and save keypoints from Mediapipe results as .npy files.
#     """
#     keypoints = extract_keypoints(results)
#     npy_path = os.path.join(DATA_PATH, action, str(sequence), f"{frame_num}.npy")
#     np.save(npy_path, keypoints)
#
# def load_data():
#     """
#     Load and preprocess the data for training.
#     """
#     sequences, labels = [], []
#     label_map = {label: num for num, label in enumerate(actions)}
#
#     for action in actions:
#         for sequence in range(no_sequences):
#             window = []
#             for frame_num in range(sequence_length):
#                 res = np.load(os.path.join(DATA_PATH, action, str(sequence), f"{frame_num}.npy"))
#                 window.append(res)
#             sequences.append(window)
#             labels.append(label_map[action])
#
#     X = np.array(sequences)
#     y = to_categorical(labels, num_classes=len(actions))
#
#     return train_test_split(X, y, test_size=0.05)
#
# def build_train_model(X_train, X_test, y_train, y_test):
#     """
#     Build and train an LSTM model on the preprocessed data.
#     """
#     log_dir = os.path.join('Logs')
#     tb_callback = TensorBoard(log_dir=log_dir)
#
#     model = Sequential([
#         LSTM(64, return_sequences=True, input_shape=(sequence_length, 1662)),
#         LSTM(128, return_sequences=True),
#         LSTM(64),
#         Dense(64, activation='relu'),
#         Dense(32, activation='relu'),
#         Dense(len(actions), activation='softmax')
#     ])
#
#     model.compile(optimizer="Adam", loss='categorical_crossentropy', metrics=['accuracy'])
#     model.fit(X_train, y_train, epochs=2000, callbacks=[tb_callback], validation_data=(X_test, y_test))
#     model.save('words_model.keras')
#     return model
#
# def extract_keypoints(results):
#     """
#     Extract keypoints from Mediapipe results for pose, face, and hands.
#     """
#     pose = np.array([[res.x, res.y, res.z, res.visibility] for res in
#                      results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 4)
#     face = np.array([[res.x, res.y, res.z] for res in
#                      results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468 * 3)
#     lh = np.array([[res.x, res.y, res.z] for res in
#                    results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
#     rh = np.array([[res.x, res.y, res.z] for res in
#                    results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(
#         21 * 3)
#     return np.concatenate([pose, face, lh, rh])
#
# def mediapipe_detection(image, model):
#     """
#     Process an image using Mediapipe model and return the processed image and results.
#     """
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # COLOR CONVERSION BGR 2 RGB
#     image.flags.writeable = False  # Image is no longer writeable
#     results = model.process(image)  # Make prediction
#     image.flags.writeable = True  # Image is now writeable
#     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # COLOR CONVERSION RGB 2 BGR
#     return image, results
#
# def draw_styled_landmarks(image, results):
#     """
#     Draw styled landmarks on the image.
#     """
#     mp_drawing = mp.solutions.drawing_utils
#     # Draw face connections
#     mp_drawing.draw_landmarks(image, results.face_landmarks, mp.solutions.holistic.FACEMESH_TESSELATION,
#                               mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
#                               mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1)
#                               )
#     # Draw pose connections
#     mp_drawing.draw_landmarks(image, results.pose_landmarks, mp.solutions.holistic.POSE_CONNECTIONS,
#                               mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
#                               mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
#                               )
#     # Draw left hand connections
#     mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp.solutions.holistic.HAND_CONNECTIONS,
#                               mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
#                               mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
#                               )
#     # Draw right hand connections
#     mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp.solutions.holistic.HAND_CONNECTIONS,
#                               mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
#                               mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
#                               )
#
# class CameraFrame(ctk.CTkFrame):
#     def __init__(self, master, cap, holistic, parent, **kwargs):
#         super().__init__(master, **kwargs)
#         self.cap = cap
#         self.holistic = holistic
#         self.parent = parent
#         self.camera_label = ctk.CTkLabel(self)
#         self.camera_label.pack(fill=ctk.BOTH, expand=True)
#         self.collect_data = False
#         self.show_camera_feed()
#
#     def show_camera_feed(self):
#         ret, frame = self.cap.read()
#         if ret:
#             image, results = mediapipe_detection(frame, self.holistic)
#             draw_styled_landmarks(image, results)
#             img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#             img = Image.fromarray(img)
#             img_tk = ImageTk.PhotoImage(image=img)
#             self.camera_label.configure(image=img_tk)
#             self.camera_label.image = img_tk
#
#             if self.collect_data:
#                 self.parent.collect_data_logic(results, frame)
#
#         self.camera_label.after(30, self.show_camera_feed)
#
#
# class SignLanguageRecognizer:
#     def __init__(self):
#         self.app = ctk.CTk()
#         self.app.title("Sign Language Data Collection and Training")
#         self.app.geometry('1024x600')  # Set window size
#
#         self.cap = cv2.VideoCapture(0)  # Changed to 0 for the default camera
#         self.holistic = mp.solutions.holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
#
#         self.camera_frame = self.create_camera_frame()
#         self.camera_frame.pack(fill=ctk.BOTH, expand=True)
#
#         # Create a frame to hold buttons
#         button_frame = ctk.CTkFrame(self.app)
#         button_frame.pack(fill=ctk.X)
#
#         self.prev_button = ctk.CTkButton(button_frame, text="PREV", command=self.train_prev_word)
#         self.prev_button.pack(side=ctk.LEFT, padx=10)
#
#         self.next_button = ctk.CTkButton(button_frame, text="NEXT", command=self.train_next_word)
#         self.next_button.pack(side=ctk.LEFT, padx=10)
#
#         self.folder_name_entry = ctk.CTkEntry(button_frame)
#         self.folder_name_entry.pack(side=ctk.LEFT, padx=10)
#
#         self.add_folder_button = ctk.CTkButton(button_frame, text="ADD", command=self.add_folder)
#         self.add_folder_button.pack(side=ctk.LEFT, padx=10)
#
#         self.start_collect_button = ctk.CTkButton(button_frame, text="START DATA COLLECTION",
#                                                   command=self.start_collection)
#         self.start_collect_button.pack(side=ctk.LEFT, padx=10)
#
#         self.start_train_button = ctk.CTkButton(button_frame, text="START TRAINING", command=self.start_training)
#         self.start_train_button.pack(side=ctk.LEFT, padx=10)  # Enabled by default
#
#         self.status_label = ctk.CTkLabel(self.app, text="")
#         self.status_label.pack(fill=ctk.X, pady=10)  # Add status label
#
#         self.current_sequence = 0
#         self.current_frame_num = 0
#
#         self.update_status()  # Initialize status label
#
#     # Existing methods ...
#
#     def create_camera_frame(self):
#         return CameraFrame(self.app, self.cap, self.holistic, self)
#
#     def start_training(self):
#         self.cap.release()
#         cv2.destroyAllWindows()
#
#         self.status_label.configure(text="Loading and preprocessing data...")
#         self.app.update_idletasks()  # Update the GUI to show the status
#
#         # Load and preprocess data
#         X_train, X_test, y_train, y_test = load_data()
#
#         self.status_label.configure(text="Building and training the model...")
#         self.app.update_idletasks()  # Update the GUI to show the status
#
#         # Build and train LSTM model
#         build_train_model(X_train, X_test, y_train, y_test)
#
#         self.status_label.configure(text="Model training completed.")
#         self.app.update_idletasks()  # Update the GUI to show the status
#
#     def collect_data_logic(self, results=None, frame=None):
#         global current_word_index
#         if current_word_index < len(actions):
#             action = actions[current_word_index]
#             if self.current_sequence < no_sequences:
#                 if self.current_frame_num < sequence_length:
#                     print(
#                         f"Collecting data for {action} - Sequence {self.current_sequence} Frame {self.current_frame_num}")
#
#                     if results is not None and frame is not None:
#                         save_keypoints(results, action, self.current_sequence, self.current_frame_num)
#
#                     self.current_frame_num += 1
#                     self.update_status()
#                 else:
#                     self.current_frame_num = 0
#                     self.current_sequence += 1
#                     self.update_status()
#             else:
#                 print(f"Data collection for {action} completed.")
#                 self.camera_frame.collect_data = False
#                 self.status_label.configure(text=f"Data collection for {action} completed. Click NEXT to continue.")
#                 self.app.update_idletasks()  # Ensure the status label updates immediately
#         else:
#             self.camera_frame.collect_data = False
#             self.status_label.configure(text="Data collection completed for all words.")
#             self.app.update_idletasks()  # Ensure the status label updates immediately
#
#     def train_next_word(self):
#         global current_word_index, DATA_PATH
#         actions = np.array(
#             [action for action in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, action))])
#         if current_word_index < len(actions) - 1:
#             current_word_index += 1
#             print(f"Training next words: {actions[current_word_index]}")
#             self.current_sequence = 0  # Reset sequence counter
#             self.current_frame_num = 0  # Reset frame counter
#             self.update_status()
#             # Do not start collecting data automatically
#         else:
#             print("No more words to train.")
#             self.status_label.configure(text="No more words to train.")
#             self.app.update_idletasks()
#
#     def train_prev_word(self):
#         global current_word_index, DATA_PATH
#         actions = np.array(
#             [action for action in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, action))])
#         if current_word_index > 0:
#             current_word_index -= 1
#             print(f"Training previous words: {actions[current_word_index]}")
#             self.current_sequence = 0  # Reset sequence counter
#             self.current_frame_num = 0  # Reset frame counter
#             self.update_status()
#             # Do not start collecting data automatically
#         else:
#             print("Already at the first words.")
#             self.status_label.configure(text="Already at the first words.")
#             self.app.update_idletasks()
#
#     def start_collection(self):
#         self.camera_frame.collect_data = True
#         self.current_sequence = 0
#         self.current_frame_num = 0
#         self.update_status()
#
#     def update_status(self):
#         global current_word_index
#         if current_word_index < len(actions):
#             action = actions[current_word_index]
#             if self.camera_frame.collect_data:
#                 self.status_label.configure(
#                     text=f"Collecting data for {action} - Sequence {self.current_sequence + 1}/{no_sequences} Frame {self.current_frame_num + 1}/{sequence_length}")
#             else:
#                 self.status_label.configure(
#                     text=f"Ready to collect data for {action}. Click START DATA COLLECTION to begin.")
#         else:
#             self.status_label.configure(text="Data collection completed for all words.")
#         self.app.update_idletasks()  # Ensure the status label updates immediately
#
#     def add_folder(self):
#         folder_name = self.folder_name_entry.get()
#         if folder_name:
#             folder_path = os.path.join(DATA_PATH, folder_name)
#             os.makedirs(folder_path, exist_ok=True)
#             for sequence in range(no_sequences):
#                 sequence_folder_path = os.path.join(folder_path, str(sequence))
#                 os.makedirs(sequence_folder_path, exist_ok=True)
#             print(f"Folder '{folder_name}' with {no_sequences} numbered folders created successfully.")
#         else:
#             print("Please enter a folder name.")
#
#
# if __name__ == "__main__":
#     current_word_index = 0
#     SignLanguageRecognizer().app.mainloop()


# TESTING WORDS
# import customtkinter as tk
# import cv2
# from PIL import Image, ImageTk
# import numpy as np
# import mediapipe as mp
# from tensorflow.keras.models import load_model
# from googletrans import Translator
# from gtts import gTTS
# import os
# from language_api import LanguageAPI
# import pygame
#
# # Global variables
# scaled_value = 0.8
# recognized_word = ""
#
#
# class CustomFrameApp:
#     def __init__(self, master):
#         self.master = master
#         self.master.title("Custom Frames")
#         self.master.geometry("1024x600")
#         self.language_helper = LanguageAPI()
#
#         self.DATA_PATH = 'MP_Data_Words'
#         self.actions = np.array(
#             [action for action in os.listdir(self.DATA_PATH) if os.path.isdir(os.path.join(self.DATA_PATH, action))])
#         self.sequence = []
#         self.model = load_model('words_model.keras')
#
#         self.setup_ui()
#         self.setup_camera()
#         self.setup_mediapipe()
#
#         self.show_camera_feed()
#
#     def setup_ui(self):
#         self.main_frame = tk.CTkFrame(self.master)
#         self.main_frame.pack(fill=tk.BOTH, expand=True)
#         self.main_frame["bg"] = "lightgray"
#
#         half_width = 1024 // 2
#
#         self.MainCamera = tk.CTkFrame(self.main_frame, width=half_width, height=600)
#         self.MainCamera.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
#         self.MainCamera["bg"] = "lightblue"
#
#         self.MainMenus = tk.CTkFrame(self.main_frame, width=half_width, height=600)
#         self.MainMenus.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
#         self.MainMenus["bg"] = "lightgreen"
#
#         self.MainMenus.pack_propagate(0)
#         self.MainCamera.pack_propagate(0)
#
#         self.setup_sentence_frame(half_width)
#         self.setup_gesture_frame(half_width)
#         self.setup_translation_frame(half_width)
#
#     def setup_sentence_frame(self, half_width):
#         self.SentenceFrame = tk.CTkFrame(self.MainMenus, width=half_width, height=1)
#         self.SentenceFrame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)
#         self.SentenceFrame["bg"] = "lightblue"
#
#         self.sentence_label = tk.CTkLabel(self.SentenceFrame, text="Sentence: ", font=("Arial", 14))
#         self.sentence_label.pack(side=tk.TOP, pady=20)
#
#         button_frame = tk.CTkFrame(self.SentenceFrame, height=70)  # Adjust the height here
#         button_frame.pack(side=tk.BOTTOM, pady=5)
#
#         self.add_word_btn = tk.CTkButton(button_frame, text="ADD WORD", command=self.add_word_to_sentence)
#         self.add_word_btn.pack(side=tk.LEFT, padx=5)
#
#         self.tts_button = tk.CTkButton(button_frame, text="SPEAK", command=self.speak_sentence)
#         self.tts_button.pack(side=tk.LEFT, padx=5)
#
#         self.clear_button = tk.CTkButton(button_frame, text="CLEAR", command=self.clear_sentence)
#         self.clear_button.pack(side=tk.LEFT, padx=5)
#
#     def setup_gesture_frame(self, half_width):
#         self.GestureFrame = tk.CTkFrame(self.MainMenus, width=half_width, height=1)
#         self.GestureFrame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)
#         self.GestureFrame["bg"] = "lightblue"
#
#         self.recognized_label = tk.CTkLabel(self.GestureFrame, text="Recognized Gesture: ", font=("Arial", 14))
#         self.recognized_label.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
#
#     def setup_translation_frame(self, half_width):
#         self.TranslationFrame = tk.CTkFrame(self.MainMenus, width=half_width, height=1)
#         self.TranslationFrame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)
#         self.TranslationFrame["bg"] = "lightblue"
#
#         self.translator = Translator()
#         self.languages = ['en', 'es', 'fr', 'de', 'it', 'zh-cn', 'ja', 'ko']
#         self.language_names = ['English', 'Spanish', 'French', 'German', 'Italian', 'Chinese', 'Japanese', 'Korean']
#
#         self.language_label = tk.CTkLabel(self.TranslationFrame, text="Select Language:", font=("Arial", 14))
#         self.language_label.pack(side=tk.LEFT, padx=10, pady=20)
#
#         self.language_var = tk.StringVar()
#         self.language_combobox = tk.CTkComboBox(self.TranslationFrame, values=self.language_helper.get_language_names(),
#                                                 variable=self.language_var)
#         self.language_combobox.pack(side=tk.LEFT, padx=10, pady=20)
#
#         self.translate_button = tk.CTkButton(self.TranslationFrame, text="TRANSLATE", command=self.translate_sentence)
#         self.translate_button.pack(side=tk.LEFT, padx=10, pady=20)
#
#     def setup_camera(self):
#         self.video_capture = cv2.VideoCapture(0)
#         self.camera_label = tk.CTkLabel(self.MainCamera)
#         self.camera_label.pack(fill=tk.BOTH, expand=True)
#
#     def setup_mediapipe(self):
#         self.mp_holistic = mp.solutions.holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
#
#     def show_camera_feed(self):
#         ret, frame = self.video_capture.read()
#         if ret:
#             frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             resized_frame = cv2.resize(frame_rgb, (self.MainCamera.winfo_width(), self.MainCamera.winfo_height()))
#             img = Image.fromarray(resized_frame)
#             img_tk = ImageTk.PhotoImage(image=img)
#
#             self.camera_label.configure(image=img_tk)
#             self.camera_label.image = img_tk
#
#             self.gesture_recognition(frame_rgb)
#
#         self.camera_label.after(30, self.show_camera_feed)
#
#     def gesture_recognition(self, frame):
#         global scaled_value, recognized_word
#         image, results = self.mediapipe_detection(frame)
#         keypoints = self.extract_keypoints(results)
#         self.sequence.append(keypoints)
#         self.sequence = self.sequence[-30:]
#
#         if len(self.sequence) == 30:
#             res = self.model.predict(np.expand_dims(self.sequence, axis=0))[0]
#             action_index = np.argmax(res)
#
#             if res[action_index] > scaled_value:
#                 recognized_word = self.actions[action_index]
#                 self.display_recognized_word(recognized_word)
#
#     def display_recognized_word(self, recognized_word):
#         self.recognized_label.configure(text=f"Recognized Gesture: {recognized_word}")
#
#     def mediapipe_detection(self, image):
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         image.flags.writeable = False
#         results = self.mp_holistic.process(image)
#         image.flags.writeable = True
#         image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#         return image, results
#
#     def extract_keypoints(self, results):
#         pose = np.array([[res.x, res.y, res.z, res.visibility] for res in
#                          results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 4)
#         face = np.array([[res.x, res.y, res.z] for res in
#                          results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468 * 3)
#         lh = np.array([[res.x, res.y, res.z] for res in
#                        results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(
#             21 * 3)
#         rh = np.array([[res.x, res.y, res.z] for res in
#                        results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(
#             21 * 3)
#         return np.concatenate([pose, face, lh, rh])
#
#     def add_word_to_sentence(self):
#         global recognized_word
#         sentence = self.sentence_label.cget("text")
#         new_sentence = f"{sentence} {recognized_word}".strip()
#         self.sentence_label.configure(text=new_sentence)
#
#     def clear_sentence(self):
#         self.sentence_label.configure(text="Sentence: ")
#
#     def speak_sentence(self):
#         sentence = self.sentence_label.cget("text")[10:]  # Remove "Sentence: " prefix
#         selected_language = self.language_var.get()
#         if selected_language:
#             target_language = self.language_helper.get_language_codes()[
#                 self.language_helper.get_language_names().index(selected_language)]
#         else:
#             # Set default target language to English
#             target_language = 'en'
#
#         tts = gTTS(text=sentence, lang=target_language)
#         tts.save("temp.mp3")
#
#         # Initialize pygame mixer
#         pygame.mixer.init()
#
#         # Load and play the audio file
#         pygame.mixer.music.load("temp.mp3")
#         pygame.mixer.music.play()
#
#         # Wait until playback finishes
#         while pygame.mixer.music.get_busy():
#             pygame.time.Clock().tick(10)
#
#         # Clean up
#         pygame.mixer.quit()
#         os.remove("temp.mp3")
#
#     def translate_sentence(self):
#         sentence = self.sentence_label.cget("text")[10:]  # Remove "Sentence: " prefix
#         selected_language = self.language_var.get()
#         if selected_language:
#             target_language = self.language_helper.get_language_codes()[
#                 self.language_helper.get_language_names().index(selected_language)]
#         else:
#             # Set default target language to English
#             target_language = 'en'
#
#         translated = self.translator.translate(sentence, dest=target_language).text
#         self.sentence_label.configure(text=f"Sentence: {translated}")
#
#
# if __name__ == "__main__":
#     root = tk.CTk()
#     app = CustomFrameApp(root)
#     root.mainloop()

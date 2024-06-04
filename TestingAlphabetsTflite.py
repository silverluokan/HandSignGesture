import customtkinter as tk
import cv2
from PIL import Image, ImageTk
import numpy as np
import mediapipe as mp
import tensorflow as tf
from googletrans import Translator
from gtts import gTTS
import os
from language_api import LanguageAPI
import pygame
from firebase import firebase

# Global variables
scaled_value = 0.5
recognized_word = ""
firebase = firebase.FirebaseApplication('https://hand-gesture-8dd5e-default-rtdb.asia-southeast1.firebasedatabase.app/')

class CustomFrameApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Custom Frames")
        self.master.geometry("1024x600")
        self.language_helper = LanguageAPI()

        self.actions = firebase.get('/alphabets_model', None)
        print(self.actions)
        self.sequence = []

        # Load the TFLite model and allocate tensors
        self.interpreter = tf.lite.Interpreter(model_path='alphabets_model.tflite')
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.setup_ui()
        self.setup_camera()
        self.setup_mediapipe()

        self.show_camera_feed()

    def setup_ui(self):
        self.main_frame = tk.CTkFrame(self.master)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        self.main_frame["bg"] = "lightgray"

        half_width = 1024 // 2

        self.MainCamera = tk.CTkFrame(self.main_frame, width=half_width, height=600)
        self.MainCamera.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.MainCamera["bg"] = "lightblue"

        self.MainMenus = tk.CTkFrame(self.main_frame, width=half_width, height=600)
        self.MainMenus.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.MainMenus["bg"] = "lightgreen"

        self.MainMenus.pack_propagate(0)
        self.MainCamera.pack_propagate(0)

        self.setup_sentence_frame(half_width)
        self.setup_gesture_frame(half_width)
        self.setup_translation_frame(half_width)

    def setup_sentence_frame(self, half_width):
        self.SentenceFrame = tk.CTkFrame(self.MainMenus, width=half_width, height=1)
        self.SentenceFrame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.SentenceFrame["bg"] = "lightblue"

        self.sentence_label = tk.CTkLabel(self.SentenceFrame, text="Sentence: ", font=("Arial", 14))
        self.sentence_label.pack(side=tk.TOP, pady=20)

        button_frame = tk.CTkFrame(self.SentenceFrame, height=70)  # Adjust the height here
        button_frame.pack(side=tk.BOTTOM, pady=5)

        self.add_word_btn = tk.CTkButton(button_frame, text="ADD WORD", command=self.add_word_to_sentence)
        self.add_word_btn.pack(side=tk.LEFT, padx=5)

        self.tts_button = tk.CTkButton(button_frame, text="SPEAK", command=self.speak_sentence)
        self.tts_button.pack(side=tk.LEFT, padx=5)

        self.clear_button = tk.CTkButton(button_frame, text="CLEAR", command=self.clear_sentence)
        self.clear_button.pack(side=tk.LEFT, padx=5)

    def setup_gesture_frame(self, half_width):
        self.GestureFrame = tk.CTkFrame(self.MainMenus, width=half_width, height=1)
        self.GestureFrame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.GestureFrame["bg"] = "lightblue"

        self.recognized_label = tk.CTkLabel(self.GestureFrame, text="Recognized Gesture: ", font=("Arial", 14))
        self.recognized_label.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def setup_translation_frame(self, half_width):
        self.TranslationFrame = tk.CTkFrame(self.MainMenus, width=half_width, height=1)
        self.TranslationFrame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.TranslationFrame["bg"] = "lightblue"

        self.translator = Translator()
        self.languages = ['en', 'es', 'fr', 'de', 'it', 'zh-cn', 'ja', 'ko']
        self.language_names = ['English', 'Spanish', 'French', 'German', 'Italian', 'Chinese', 'Japanese', 'Korean']

        self.language_label = tk.CTkLabel(self.TranslationFrame, text="Select Language:", font=("Arial", 14))
        self.language_label.pack(side=tk.LEFT, padx=10, pady=20)

        self.language_var = tk.StringVar()
        self.language_combobox = tk.CTkComboBox(self.TranslationFrame, values=self.language_helper.get_language_names(),
                                                variable=self.language_var)
        self.language_combobox.pack(side=tk.LEFT, padx=10, pady=20)

        self.translate_button = tk.CTkButton(self.TranslationFrame, text="TRANSLATE", command=self.translate_sentence)
        self.translate_button.pack(side=tk.LEFT, padx=10, pady=20)

    def setup_camera(self):
        self.video_capture = cv2.VideoCapture(0)
        self.camera_label = tk.CTkLabel(self.MainCamera)
        self.camera_label.pack(fill=tk.BOTH, expand=True)

    def setup_mediapipe(self):
        self.mp_holistic = mp.solutions.holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def show_camera_feed(self):
        ret, frame = self.video_capture.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resized_frame = cv2.resize(frame_rgb, (self.MainCamera.winfo_width(), self.MainCamera.winfo_height()))
            img = Image.fromarray(resized_frame)
            img_tk = ImageTk.PhotoImage(image=img)

            self.camera_label.configure(image=img_tk)
            self.camera_label.image = img_tk

            self.gesture_recognition(frame_rgb)

        self.camera_label.after(30, self.show_camera_feed)

    def gesture_recognition(self, frame):
        global scaled_value, recognized_word
        image, results = self.mediapipe_detection(frame)
        keypoints = self.extract_keypoints(results)
        self.sequence.append(keypoints)
        self.sequence = self.sequence[-30:]

        if len(self.sequence) == 30:
            input_data = np.expand_dims(self.sequence, axis=0).astype(np.float32)
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            self.interpreter.invoke()
            res = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
            action_index = np.argmax(res)

            if res[action_index] > scaled_value:
                recognized_word = self.actions[action_index]
                self.display_recognized_word(recognized_word)

    def display_recognized_word(self, recognized_word):
        self.recognized_label.configure(text=f"Recognized Gesture: {recognized_word}")

    def mediapipe_detection(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = self.mp_holistic.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image, results

    def extract_keypoints(self, results):
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in
                         results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 4)
        face = np.array([[res.x, res.y, res.z] for res in
                         results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468 * 3)
        lh = np.array([[res.x, res.y, res.z] for res in
                       results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(
            21 * 3)
        rh = np.array([[res.x, res.y, res.z] for res in
                       results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(
            21 * 3)
        return np.concatenate([pose, face, lh, rh])

    def add_word_to_sentence(self):
        global recognized_word
        sentence = self.sentence_label.cget("text")
        new_sentence = f"{sentence}{recognized_word}".strip()
        self.sentence_label.configure(text=new_sentence)

    def clear_sentence(self):
        self.sentence_label.configure(text="Sentence: ")

    def speak_sentence(self):
        sentence = self.sentence_label.cget("text")[10:]  # Remove "Sentence: " prefix
        selected_language = self.language_var.get()
        if selected_language:
            target_language = self.language_helper.get_language_codes()[
                self.language_helper.get_language_names().index(selected_language)]
        else:
            # Set default target language to English
            target_language = 'en'

        tts = gTTS(text=sentence, lang=target_language)
        tts.save("temp.mp3")

        # Initialize pygame mixer
        pygame.mixer.init()

        # Load and play the audio file
        pygame.mixer.music.load("temp.mp3")
        pygame.mixer.music.play()

        # Wait until playback finishes
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)

        # Clean up
        pygame.mixer.quit()
        os.remove("temp.mp3")

    def translate_sentence(self):
        sentence = self.sentence_label.cget("text")[10:]  # Remove "Sentence: " prefix
        selected_language = self.language_var.get()
        if selected_language:
            target_language = self.language_helper.get_language_codes()[
                self.language_helper.get_language_names().index(selected_language)]
        else:
            # Set default target language to English
            target_language = 'en'

        translated = self.translator.translate(sentence, dest=target_language).text
        self.sentence_label.configure(text=f"Sentence: {translated}")


if __name__ == "__main__":
    root = tk.CTk()
    app = CustomFrameApp(root)
    root.mainloop()

#
# import customtkinter as tk
# import cv2
# from PIL import Image, ImageTk
# import numpy as np
# import mediapipe as mp
# import tensorflow as tf
# from googletrans import Translator
# from gtts import gTTS
# import os
# from language_api import LanguageAPI
# import pygame
# import RPi.GPIO as GPIO
#
# # Global variables
# scaled_value = 0.8
# recognized_word = ""
#
# # GPIO setup
# GPIO.setmode(GPIO.BCM)
# SPEAK_PIN = 0
# CLEAR_PIN = 2
# ADD_WORD_PIN = 3
#
# GPIO.setup(SPEAK_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
# GPIO.setup(CLEAR_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
# GPIO.setup(ADD_WORD_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
#
#
# class CustomFrameApp:
#     def __init__(self, master):
#         self.master = master
#         self.master.title("Custom Frames")
#         self.master.geometry("1024x600")
#         self.language_helper = LanguageAPI()
#
#         self.DATA_PATH = 'MP_Data_Alphabet'
#         self.actions = np.array(
#             [action for action in os.listdir(self.DATA_PATH) if os.path.isdir(os.path.join(self.DATA_PATH, action))])
#         self.sequence = []
#
#         # Load the TFLite model and allocate tensors
#         self.interpreter = tf.lite.Interpreter(model_path='alphabets_model.tflite')
#         self.interpreter.allocate_tensors()
#         self.input_details = self.interpreter.get_input_details()
#         self.output_details = self.interpreter.get_output_details()
#
#         self.setup_ui()
#         self.setup_camera()
#         self.setup_mediapipe()
#
#         # Setup GPIO event detection
#         GPIO.add_event_detect(SPEAK_PIN, GPIO.FALLING, callback=self.speak_sentence, bouncetime=300)
#         GPIO.add_event_detect(CLEAR_PIN, GPIO.FALLING, callback=self.clear_sentence, bouncetime=300)
#         GPIO.add_event_detect(ADD_WORD_PIN, GPIO.FALLING, callback=self.add_word_to_sentence, bouncetime=300)
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
#             input_data = np.expand_dims(self.sequence, axis=0).astype(np.float32)
#             self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
#             self.interpreter.invoke()
#             res = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
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
#     def add_word_to_sentence(self, channel=None):
#         global recognized_word
#         sentence = self.sentence_label.cget("text")
#         new_sentence = f"{sentence}{recognized_word}".strip()
#         self.sentence_label.configure(text=new_sentence)
#
#     def clear_sentence(self, channel=None):
#         self.sentence_label.configure(text="Sentence: ")
#
#     def speak_sentence(self, channel=None):
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
#     GPIO.cleanup()  # Clean up GPIO pins on exit
#


# import customtkinter as tk
# import cv2
# from PIL import Image, ImageTk
# import numpy as np
# import mediapipe as mp
# import tensorflow as tf
# from googletrans import Translator
# from gtts import gTTS
# import os
# from language_api import LanguageAPI
# import pygame
# import RPi.GPIO as GPIO
#
# # Global variables
# scaled_value = 0.8
# recognized_word = ""
#
# # GPIO setup
# GPIO.setmode(GPIO.BCM)
# SPEAK_PIN = 0
# CLEAR_PIN = 2
# ADD_WORD_PIN = 3
# LANG_UP_PIN = 4
# LANG_DOWN_PIN = 5
#
# GPIO.setup(SPEAK_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
# GPIO.setup(CLEAR_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
# GPIO.setup(ADD_WORD_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
# GPIO.setup(LANG_UP_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
# GPIO.setup(LANG_DOWN_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
#
#
# class CustomFrameApp:
#     def __init__(self, master):
#         self.master = master
#         self.master.title("Custom Frames")
#         self.master.geometry("1024x600")
#         self.language_helper = LanguageAPI()
#
#         self.DATA_PATH = 'MP_Data_Alphabet'
#         self.actions = np.array(
#             [action for action in os.listdir(self.DATA_PATH) if os.path.isdir(os.path.join(self.DATA_PATH, action))])
#         self.sequence = []
#
#         # Load the TFLite model and allocate tensors
#         self.interpreter = tf.lite.Interpreter(model_path='alphabets_model.tflite')
#         self.interpreter.allocate_tensors()
#         self.input_details = self.interpreter.get_input_details()
#         self.output_details = self.interpreter.get_output_details()
#
#         self.setup_ui()
#         self.setup_camera()
#         self.setup_mediapipe()
#
#         # Setup GPIO event detection
#         GPIO.add_event_detect(SPEAK_PIN, GPIO.FALLING, callback=self.speak_sentence, bouncetime=300)
#         GPIO.add_event_detect(CLEAR_PIN, GPIO.FALLING, callback=self.clear_sentence, bouncetime=300)
#         GPIO.add_event_detect(ADD_WORD_PIN, GPIO.FALLING, callback=self.add_word_to_sentence, bouncetime=300)
#         GPIO.add_event_detect(LANG_UP_PIN, GPIO.FALLING, callback=self.language_up, bouncetime=300)
#         GPIO.add_event_detect(LANG_DOWN_PIN, GPIO.FALLING, callback=self.language_down, bouncetime=300)
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
#             input_data = np.expand_dims(self.sequence, axis=0).astype(np.float32)
#             self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
#             self.interpreter.invoke()
#             res = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
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
#     def speak_sentence(self, channel):
#         sentence = self.sentence_label.cget("text")[10:]
#         if sentence:
#             tts = gTTS(text=sentence, lang=self.language_helper.get_language_code(self.language_combobox.get()))
#             tts.save("sentence.mp3")
#             pygame.mixer.init()
#             pygame.mixer.music.load("sentence.mp3")
#             pygame.mixer.music.play()
#
#     def clear_sentence(self, channel):
#         self.sentence_label.configure(text="Sentence: ")
#
#     def add_word_to_sentence(self, channel):
#         global recognized_word
#         current_sentence = self.sentence_label.cget("text")[10:]
#         if recognized_word:
#             new_sentence = current_sentence + " " + recognized_word if current_sentence else recognized_word
#             self.sentence_label.configure(text="Sentence: " + new_sentence)
#
#     def language_up(self, channel):
#         current_index = self.language_names.index(self.language_combobox.get())
#         new_index = (current_index + 1) % len(self.language_names)
#         self.language_combobox.set(self.language_names[new_index])
#
#     def language_down(self, channel):
#         current_index = self.language_names.index(self.language_combobox.get())
#         new_index = (current_index - 1) % len(self.language_names)
#         self.language_combobox.set(self.language_names[new_index])
#
#     def translate_sentence(self):
#         sentence = self.sentence_label.cget("text")[10:]
#         if sentence:
#             target_lang = self.language_helper.get_language_code(self.language_combobox.get())
#             translated = self.translator.translate(sentence, dest=target_lang).text
#             self.sentence_label.configure(text="Sentence: " + translated)
#
#
# root = tk.CTk()
# app = CustomFrameApp(root)
# root.mainloop()
#

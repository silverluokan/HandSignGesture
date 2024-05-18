import customtkinter as tk
import cv2
from PIL import Image, ImageTk
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import pyttsx3

scaled_value = 0.5
recognized_word = ""

class CustomFrameApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Custom Frames")
        self.master.geometry("1024x600")

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

        label_menu = tk.CTkLabel(self.MainMenus, text="Main Menu", font=("Arial", 32))
        label_menu.pack(pady=10)

        self.ThresholdFrame = tk.CTkFrame(self.MainMenus, width=half_width, height=1)
        self.ThresholdFrame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.ThresholdFrame["bg"] = "lightblue"

        self.SentenceFrame = tk.CTkFrame(self.MainMenus, width=half_width, height=1)
        self.SentenceFrame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.SentenceFrame["bg"] = "lightblue"

        self.GestureFrame = tk.CTkFrame(self.MainMenus, width=half_width, height=1)
        self.GestureFrame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.GestureFrame["bg"] = "lightblue"

        self.TTSFrame = tk.CTkFrame(self.MainMenus, width=half_width, height=1)
        self.TTSFrame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.TTSFrame["bg"] = "lightblue"

        label_threshold = tk.CTkLabel(self.ThresholdFrame, text="THRESHOLD", font=("Arial", 14))
        label_threshold.pack(side=tk.LEFT, padx=10, pady=20)

        self.slider_var = tk.DoubleVar()
        slider = tk.CTkSlider(self.ThresholdFrame, from_=0, to=100, variable=self.slider_var)
        slider.pack(side=tk.LEFT, padx=10, pady=20)

        self.scaled_label = tk.CTkLabel(self.ThresholdFrame, text="Scaled Value: 0.0", font=("Arial", 12))
        self.scaled_label.pack(side=tk.LEFT, padx=10, pady=20)

        self.slider_var.trace_add("write", self.update_slider_value)

        self.video_capture = cv2.VideoCapture(0)

        self.camera_label = tk.CTkLabel(self.MainCamera)
        self.camera_label.pack(fill=tk.BOTH, expand=True)

        self.model = load_model('action_recognition_model.keras')
        self.mp_holistic = mp.solutions.holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.actions = np.array(['HELLO', 'PLEASE', 'WELCOME', 'YES', 'DEAF', 'FOOD', 'THANK YOU'])

        self.sequence = []

        self.sentence_label = tk.CTkLabel(self.SentenceFrame, text="Sentence: ", font=("Arial", 14))
        self.sentence_label.pack(side=tk.TOP, pady=20)

        self.add_word_btn = tk.CTkButton(self.SentenceFrame, text="ADD WORD", command=self.add_word_to_sentence)
        self.add_word_btn.pack(side=tk.BOTTOM, pady=25)

        self.recognized_label = tk.CTkLabel(self.GestureFrame, text="Recognized Gesture: ", font=("Arial", 14))
        self.recognized_label.pack()

        self.tts_button = tk.CTkButton(self.TTSFrame, text="SPEAK", command=self.speak_sentence)
        self.tts_button.pack(pady=20)

        self.engine = pyttsx3.init()

        self.show_camera_feed()

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
            res = self.model.predict(np.expand_dims(self.sequence, axis=0))[0]
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
                        results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
        rh = np.array([[res.x, res.y, res.z] for res in
                        results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)
        return np.concatenate([pose, face, lh, rh])

    def add_word_to_sentence(self):
        global recognized_word
        sentence = self.sentence_label.cget("text")
        new_sentence = sentence + " " + recognized_word
        self.sentence_label.configure(text=new_sentence)

    def speak_sentence(self):
        sentence = self.sentence_label.cget("text")
        self.engine.say(sentence)
        self.engine.runAndWait()

    def update_slider_value(self, *args):
        global scaled_value
        scaled_value = self.slider_var.get() / 100.0
        self.scaled_label.configure(text=f"Scaled Value: {scaled_value:.2f}")

if __name__ == "__main__":
    root = tk.CTk()
    app = CustomFrameApp(root)
    root.mainloop()

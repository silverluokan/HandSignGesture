import customtkinter as tk
import cv2
from PIL import Image, ImageTk
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
scaled_value = 0.9
class CustomFrameApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Custom Frames")
        self.master.geometry("1024x600")  # Set root window size to 1024x600 pixels

        # Create the main frame
        self.main_frame = tk.CTkFrame(self.master)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        self.main_frame["bg"] = "lightgray"  # Set background color using dictionary syntax

        # Calculate the width for both frames (half of the window width)
        half_width = 1024 // 2

        # Create the big frame on the left
        self.MainCamera = tk.CTkFrame(self.main_frame, width=half_width, height=600)
        self.MainCamera.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.MainCamera["bg"] = "lightblue"  # Set background color using dictionary syntax

        # Create another big frame on the right
        self.MainMenus = tk.CTkFrame(self.main_frame, width=half_width, height=600)
        self.MainMenus.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.MainMenus["bg"] = "lightgreen"  # Set background color using dictionary syntax

        # Prevent MainMenus from automatically resizing based on contents
        self.MainMenus.pack_propagate(0)
        self.MainCamera.pack_propagate(0)

        # Create a label "Main Menu" inside MainMenus
        label_menu = tk.CTkLabel(self.MainMenus, text="Main Menu", font=("Arial", 32))
        label_menu.pack(pady=10)

        # Create a label "THRESHOLD" above the slider
        label_threshold = tk.CTkLabel(self.MainMenus, text="THRESHOLD", font=("Arial", 14))
        label_threshold.pack(pady=(20, 5))  # Padding added to adjust spacing

        # Create a scale (slider) between 0 and 100
        self.slider_var = tk.DoubleVar()
        slider = tk.CTkSlider(self.MainMenus, from_=0, to=100, variable=self.slider_var)
        slider.pack(pady=5)  # Adjusted padding after the label

        # Create a label to display scaled value
        self.scaled_label = tk.CTkLabel(self.MainMenus, text="Scaled Value: 0.0", font=("Arial", 12))
        self.scaled_label.pack(pady=(5, 10))  # Padding added to adjust spacing

        # Set a callback function to update the slider value between 0 and 1 with one decimal point
        self.slider_var.trace_add("write", self.update_slider_value)

        # Initialize camera capture
        self.video_capture = cv2.VideoCapture(0)  # Change the index if using a different camera

        # Create a label for displaying camera feed
        self.camera_label = tk.CTkLabel(self.MainCamera)
        self.camera_label.pack(fill=tk.BOTH, expand=True)

        # Load saved model and other required objects
        self.model = load_model('action_recognition_model.keras')
        self.mp_holistic = mp.solutions.holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.actions = np.array(['HELLO', 'PLEASE', 'GOOD MORNING', 'HELP', 'ALL DONE', 'AGAIN', 'THANK YOU'])

        # Initialize gesture recognition sequence
        self.sequence = []

        # Create a label for displaying recognized gestures
        self.recognized_label = tk.CTkLabel(self.MainMenus, text="Recognized Gesture: ", font=("Arial", 14))
        self.recognized_label.pack(pady=(20, 5))  # Padding added to adjust spacing

        # Call the function to start displaying the camera feed
        self.show_camera_feed()

    def show_camera_feed(self):
        ret, frame = self.video_capture.read()  # Read a frame from the camera
        if ret:
            # Convert the frame from BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Resize the frame to fit into the camera label
            resized_frame = cv2.resize(frame_rgb, (self.MainCamera.winfo_width(), self.MainCamera.winfo_height()))
            # Convert the frame to a format compatible with Tkinter
            img = Image.fromarray(resized_frame)
            img_tk = ImageTk.PhotoImage(image=img)

            # Update the camera label with the new frame
            self.camera_label.configure(image=img_tk)
            self.camera_label.image = img_tk  # Keep a reference to prevent garbage collection

            # Perform gesture recognition
            self.gesture_recognition(frame_rgb)

        # Schedule the function to be called again after a delay (30 milliseconds)
        self.camera_label.after(30, self.show_camera_feed)

    def gesture_recognition(self, frame):
        global scaled_value
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
        # Update label text to display the recognized word in your Tkinter UI
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
                        results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(
            21 * 3)
        return np.concatenate([pose, face, lh, rh])

    def update_slider_value(self, *args):
        global scaled_value
        # Update the slider value between 0 and 1 with one decimal point
        scaled_value = self.slider_var.get() / 100.0  # Scale the value to be between 0 and 1
        self.scaled_label.configure(text=f"Scaled Value: {scaled_value:.1f}")  # Update label text using configure
        print(f"Scaled Value: {scaled_value:.1f}")


if __name__ == "__main__":
    root = tk.CTk()
    app = CustomFrameApp(root)
    root.mainloop()
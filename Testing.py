import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import tensorflow as tf

# Load saved model
model = load_model('action_recognition_model.keras')

# Load Mediapipe and define necessary functions
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils


def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
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


def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        prob_scalar = prob.item() if isinstance(prob, np.ndarray) else prob
        cv2.rectangle(output_frame, (0, 60 + num * 40), (int(prob_scalar * 100), 90 + num * 40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85 + num * 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                    cv2.LINE_AA)
    return output_frame


def display_gesture_sentence(image, sentence):
    cv2.rectangle(image, (0, 0), (640, 40), (0, 0, 0), -1)
    cv2.putText(image, ' '.join(sentence), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    return image


# Real-time gesture recognition loop
def gesture_recognition(cap, holistic, model):
    sequence = []
    recognized_word = ""  # Variable to hold recognized word
    # recognize_word = False  # Flag to trigger word recognition

    while cap.isOpened():
        ret, frame = cap.read()
        image, results = mediapipe_detection(frame, holistic)

        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-30:]

        # Check for key press to trigger word recognition
        key = cv2.waitKey(10)
        # if key & 0xFF == ord('a'):  # Check for 'A' key press
        #     recognize_word = True

        if len(sequence) == 30:  # and recognize_word: Check if word recognition is triggered
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            action_index = np.argmax(res)

            if res[action_index] > 0.5:
                action = actions[action_index]
                recognized_word = action
                # recognize_word = False  # Reset flag after word recognition

                # Display video frame with recognized word as subtitle
            subtitle_text = f"Recognized Word: {recognized_word}"
            text_size, _ = cv2.getTextSize(subtitle_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            text_x = int((image.shape[1] - text_size[0]) / 2)  # Center text horizontally
            text_y = image.shape[0] - 20  # Position text near bottom
            cv2.putText(image, subtitle_text, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow('OpenCV Feed', image)

        if key & 0xFF == ord('q'):  # Check for 'Q' key press to quit
            break

    cap.release()
    cv2.destroyAllWindows()


# Main testing program
if __name__ == "__main__":
    cap = cv2.VideoCapture(1)
    holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    actions = np.array(['HELLO', 'PLEASE', 'GOOD MORNING', 'HELP', 'ALL DONE', 'AGAIN', 'THANK YOU'])

    gesture_recognition(cap, holistic, model)

    cap.release()
    cv2.destroyAllWindows()

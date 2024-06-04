import customtkinter as tk
import subprocess
import os
import sys


class App:
    def __init__(self, master):
        self.master = master
        self.master.title("Gesture Recognition App")
        self.master.geometry("330x250")  # Adjusted width for side-by-side layout

        # Custom frame with blue background and rounded corners
        self.welcome_frame = tk.CTkFrame(self.master)
        self.welcome_frame.pack(fill=tk.X, padx=10, pady=10)  # Adjusted padding

        # Label for the welcome message inside the custom frame
        self.welcome_label = tk.CTkLabel(
            self.welcome_frame,
            text="WELCOME TO GESTURE APPLICATION",
            font=("Helvetica", 16),
            text_color="white",
        )
        self.welcome_label.pack(fill=tk.X, pady=5)  # Adjusted padding

        # Converted buttons to labels
        self.test_button_alphabet = tk.CTkLabel(
            self.master, text="Run Testing Alphabet Script", font=("Helvetica", 12), bg_color="green", corner_radius=5
        )
        self.test_button_alphabet.pack(fill=tk.X, padx=50, pady=35)  # Reduced padding

        self.test_button_word = tk.CTkLabel(
            self.master, text="Run Testing Word Script", font=("Helvetica", 12), bg_color="green", corner_radius=5
        )
        self.test_button_word.pack(fill=tk.X, padx=50, pady=0)  # Reduced padding

        # Bind keyboard events to functions
        self.master.bind('1', lambda event: self.run_testing_alphabet())
        self.master.bind('2', lambda event: self.run_testing_word())

    def run_testing_alphabet(self):
        self.run_script('TestingAlphabetsTflite.py')

    def run_testing_word(self):
        self.run_script('TestingWordTflite.py')

    def run_script(self, script_name):
        # Print current working directory and python executable for debugging
        print("Current Working Directory:", os.getcwd())
        print("Python Executable:", sys.executable)

        # Ensure that the script is run with the correct python interpreter
        script_path = os.path.join(os.getcwd(), script_name)
        subprocess.Popen([sys.executable, script_path])


if __name__ == "__main__":
    root = tk.CTk()
    app = App(root)
    root.mainloop()

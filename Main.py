import customtkinter as tk
import subprocess
import os
import sys


class App:
    def __init__(self, master):
        self.master = master
        self.master.title("Gesture Recognition App")
        self.master.geometry("300x250")

        self.frame = tk.CTkFrame(self.master)
        self.frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        self.test_button = tk.CTkButton(self.frame, text="Run Testing Alphabet Script", command=self.run_testing_alphabet)
        self.test_button.pack(pady=10)

        self.test_button = tk.CTkButton(self.frame, text="Run Testing Word Script", command=self.run_testing_word)
        self.test_button.pack(pady=10)

    def run_testing_alphabet(self):
        self.run_script('TestingAlphabet.py')

    def run_testing_word(self):
        self.run_script('TestingWord.py')

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

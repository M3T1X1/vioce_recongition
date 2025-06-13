import os.path
import tkinter as tk
from os.path import exists
from tkinter import filedialog, messagebox
import librosa
import numpy as np
from scipy.spatial.distance import euclidean
import warnings
import pandas as pd
import matplotlib.pyplot as plt


warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def extract_mfcc_mean(audio_path, n_mfcc=13):
    y, sr = librosa.load(audio_path, sr=16000, mono=True)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_mean = np.mean(mfcc, axis=1)
    return mfcc_mean

def compare_voices(file1, file2):
    mfcc1 = extract_mfcc_mean(file1)
    mfcc2 = extract_mfcc_mean(file2)
    distance = euclidean(mfcc1, mfcc2)
    return distance

def chart_generation(file1, file2):
    fig, axes = plt.subplots(2, 2, figsize=(12, 6))
    files = [file1, file2]
    titles = ["Plik 1", "Plik 2"]

    for i, file in enumerate(files):
        y, sr = librosa.load(file, sr=16000, mono=True)

        # Wykres amplitudy (sygnał czasowy)
        axes[0][i].plot(np.linspace(0, len(y)/sr, len(y)), y)
        axes[0][i].set_title(f"{titles[i]} – Amplituda w czasie")
        axes[0][i].set_xlabel("Czas (s)")
        axes[0][i].set_ylabel("Amplituda")

        # Wykres spektrogramu (częstotliwości)
        S = librosa.stft(y)
        S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
        img = librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='hz', ax=axes[1][i])
        axes[1][i].set_title(f"{titles[i]} – Spektrogram (Hz)")

    plt.tight_layout()

    if not os.path.exists('wykresy'):
        os.mkdir('wykresy')

    path = os.path.join("wykresy","wykres.pdf")
    plt.savefig(path)

    plt.show()

class VoiceComparerApp:
    def __init__(self, root):
        self.root = root
        self.root.geometry("400x300")
        self.root.title("Porównywarka głosów")
        self.file1 = None
        self.file2 = None

        self.label1 = tk.Label(root, text="Plik 1: (nie wybrano)")
        self.label1.pack(pady=5)

        self.btn1 = tk.Button(root, text="Wybierz plik 1", command=self.choose_file1)
        self.btn1.pack(pady=5)

        self.label2 = tk.Label(root, text="Plik 2: (nie wybrano)")
        self.label2.pack(pady=5)

        self.btn2 = tk.Button(root, text="Wybierz plik 2", command=self.choose_file2)
        self.btn2.pack(pady=5)

        self.compare_btn = tk.Button(root, text="Porównaj głosy", command=self.compare)
        self.compare_btn.pack(pady=10)

        self.result_label = tk.Label(root, text="", font=("Arial", 12, "bold"))
        self.result_label.pack(pady=10)

    def choose_file1(self):
        self.file1 = filedialog.askopenfilename(filetypes=[("Pliki audio", "*.wav *.mp3 *.m4a")])
        if self.file1:
            self.label1.config(text=f"Plik 1: {self.file1.split('/')[-1]}")

    def choose_file2(self):
        self.file2 = filedialog.askopenfilename(filetypes=[("Pliki audio", "*.wav *.mp3 *.m4a")])
        if self.file2:
            self.label2.config(text=f"Plik 2: {self.file2.split('/')[-1]}")

    def compare(self):
        if not self.file1 or not self.file2:
            messagebox.showwarning("Błąd", "Wybierz oba pliki audio.")
            return
        try:
            dist = compare_voices(self.file1, self.file2)
            threshold = 50  # Próg do dostosowania
            result = f"Odległość MFCC: {dist:.2f}\n"
            if dist < threshold:
                result += "Głosy prawdopodobnie należą do tej samej osoby."
            else:
                result += "To różni mówcy."
            self.result_label.config(text=result)
            chart_generation(file1=self.file1, file2=self.file2)
        except Exception as e:
            messagebox.showerror("Błąd przetwarzania", str(e))



if __name__ == "__main__":
    root = tk.Tk()
    app = VoiceComparerApp(root)
    root.mainloop()
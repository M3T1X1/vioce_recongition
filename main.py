import librosa
import numpy as np
from scipy.spatial.distance import euclidean

def extract_mfcc_mean(audio_path, n_mfcc=13):
    # Wczytaj plik audio (mono, sr=16000)
    y, sr = librosa.load(audio_path, sr=16000, mono=True)
    # Wyodrębnij MFCC (domyślnie 13 współczynników)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    # Uśrednij MFCC po osi czasowej (kolumny)
    mfcc_mean = np.mean(mfcc, axis=1)
    return mfcc_mean

def compare_voices(file1, file2):
    mfcc1 = extract_mfcc_mean(file1)
    mfcc2 = extract_mfcc_mean(file2)
    # Oblicz odległość euklidesową
    distance = euclidean(mfcc1, mfcc2)
    return distance

if __name__ == "__main__":
    # Ścieżki do plików audio dwóch mówców (np. .wav)
    file1 = "Nagranie.m4a"
    file2 = "Nagranie (2).m4a"

    # Wyodrębnij i porównaj głosy
    dist = compare_voices(file1, file2)
    print(f"Odległość euklidesowa MFCC między plikami: {dist:.2f}")

    # Próg można ustalić eksperymentalnie – im mniejsza odległość, tym bardziej podobne głosy
    threshold = 50  # Przykładowa wartość, do dostosowania
    if dist < threshold:
        print("Głosy prawdopodobnie należą do tej samej osoby.")
    else:
        print("To różni mówcy.")
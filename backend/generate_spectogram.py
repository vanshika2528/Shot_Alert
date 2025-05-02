import sys
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

def generate_spectrogram(audio_file, save_path):
    y, sr = librosa.load(audio_file, sr=22050)
    spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
    spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)

    plt.figure(figsize=(2.24, 2.24), dpi=100)  # Size adjusted for 224x224 output
    librosa.display.specshow(spectrogram_db, sr=sr, x_axis='time', y_axis='mel')
    plt.axis('off')  # No axis
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

if __name__ == "__main__":
    audio_path = sys.argv[1]
    save_path = sys.argv[2]
    generate_spectrogram(audio_path, save_path)

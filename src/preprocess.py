import librosa
import soundfile as sf
import os

INPUT_PATH = "data/original_segment.wav"
OUTPUT_PATH = "data/clean.wav"

def preprocess_audio(input_path, output_path):
    print("Loading audio...")
    y, sr = librosa.load(input_path, sr=16000)

    print(f"Original Sample Rate: {sr}")
    print(f"Audio Duration: {len(y)/sr:.2f} sec")

    # Normalize audio
    y = librosa.util.normalize(y)

    # Simple noise reduction (pre-emphasis)
    y = librosa.effects.preemphasis(y)

    sf.write(output_path, y, sr)
    print(f"Saved cleaned audio to {output_path}")

if __name__ == "__main__":
    preprocess_audio(INPUT_PATH, OUTPUT_PATH)
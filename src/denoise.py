import librosa
import numpy as np
import soundfile as sf

INPUT_PATH = "data/original_segment.wav"
OUTPUT_PATH = "data/denoised.wav"


def spectral_subtraction(y, sr):
    # STFT
    D = librosa.stft(y)
    magnitude, phase = np.abs(D), np.angle(D)

    # Estimate noise from first 0.5 sec
    noise_frames = int(0.5 * sr / 512)
    noise_mag = np.mean(magnitude[:, :noise_frames], axis=1, keepdims=True)

    # Subtract noise
    clean_mag = magnitude - noise_mag
    clean_mag = np.maximum(clean_mag, 0.0)

    # Reconstruct
    clean_stft = clean_mag * np.exp(1j * phase)
    clean_audio = librosa.istft(clean_stft)

    return clean_audio


def main():
    print("Loading audio...")
    y, sr = librosa.load(INPUT_PATH, sr=16000)

    print("Applying spectral subtraction...")
    clean = spectral_subtraction(y, sr)

    print("Saving denoised audio...")
    sf.write(OUTPUT_PATH, clean, sr)

    print("Done!")


if __name__ == "__main__":
    main()
import torch
import librosa
import numpy as np
from lid_model import LIDModel

MODEL_PATH = "models/lid_model.pth"
AUDIO_PATH = "data/clean.wav"

MAX_LEN = 200
CHUNK_DURATION = 0.5  # in seconds


def extract_mfcc_chunk(y, sr):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T

    if len(mfcc) < MAX_LEN:
        mfcc = np.pad(mfcc, ((0, MAX_LEN - len(mfcc)), (0, 0)))
    else:
        mfcc = mfcc[:MAX_LEN]

    return mfcc


def load_model():
    model = LIDModel()
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    return model


def run_lid():
    y, sr = librosa.load(AUDIO_PATH, sr=16000)
    model = load_model()

    chunk_size = int(CHUNK_DURATION * sr)
    results = []

    for i in range(0, len(y), chunk_size):
        chunk = y[i:i + chunk_size]

        if len(chunk) < int(0.3 * sr):  
            continue

        mfcc = extract_mfcc_chunk(chunk, sr)
        mfcc = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            output = model(mfcc)
            pred = torch.argmax(output, dim=1).item()

        label = "English" if pred == 0 else "Hindi"

        start = i / sr
        end = (i + chunk_size) / sr

        results.append((start, end, label))

    return results


def smooth_predictions(results):
    smoothed = []

    for i in range(len(results)):
        labels = []

        for j in range(max(0, i - 1), min(len(results), i + 2)):
            labels.append(results[j][2])

        label = max(set(labels), key=labels.count)
        smoothed.append((results[i][0], results[i][1], label))

    return smoothed


if __name__ == "__main__":
    results = run_lid()
    results = smooth_predictions(results)

    print("\n--- LID OUTPUT (first 30 segments) ---\n")

    for r in results[:30]:
        print(f"{r[0]:.2f} - {r[1]:.2f} sec → {r[2]}")
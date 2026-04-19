import torch
import librosa
import numpy as np
from lid_model import LIDModel

AUDIO_PATH = "data/clean.wav"

def extract_mfcc(y, sr):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return mfcc.T

def main():
    y, sr = librosa.load(AUDIO_PATH, sr=16000)

    # take small segment
    y = y[:sr*5]

    mfcc = extract_mfcc(y, sr)
    mfcc = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0)
    mfcc.requires_grad = True

    model = LIDModel()
    model.load_state_dict(torch.load("models/lid_model.pth"))
    model.eval()

    output = model(mfcc)
    label = torch.argmax(output)

    loss = output[0, label]
    loss.backward()

    epsilon = 0.01
    perturbed = mfcc + epsilon * mfcc.grad.sign()

    new_output = model(perturbed)
    new_label = torch.argmax(new_output)

    print("Original label:", label.item())
    print("After attack:", new_label.item())

if __name__ == "__main__":
    main()
import librosa
import numpy as np
import torch
import torch.nn as nn

REAL_AUDIO = "data/student_voice_ref.wav"
FAKE_AUDIO = "outputs/output_cloned.wav"

def extract_features(file):
    y, sr = librosa.load(file, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfcc, axis=1)

class SpoofModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(13, 2)

    def forward(self, x):
        return self.fc(x)

def main():
    real_feat = extract_features(REAL_AUDIO)
    fake_feat = extract_features(FAKE_AUDIO)

    X = torch.tensor([real_feat, fake_feat], dtype=torch.float32)
    y = torch.tensor([0, 1])  # 0 = real, 1 = fake

    model = SpoofModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # train quickly
    for _ in range(100):
        outputs = model(X)
        loss = criterion(outputs, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    preds = torch.argmax(model(X), dim=1)

    print("\nPredictions:")
    print("Real audio →", "Real" if preds[0] == 0 else "Fake")
    print("Cloned audio →", "Fake" if preds[1] == 1 else "Real")

if __name__ == "__main__":
    main()
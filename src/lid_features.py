import librosa
import numpy as np

MAX_LEN = 200  

def extract_mfcc(file_path):
    y, sr = librosa.load(file_path, sr=16000)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc = mfcc.T

    # Padding / truncating
    if len(mfcc) < MAX_LEN:
        pad_width = MAX_LEN - len(mfcc)
        mfcc = np.pad(mfcc, ((0, pad_width), (0, 0)))
    else:
        mfcc = mfcc[:MAX_LEN]

    return mfcc
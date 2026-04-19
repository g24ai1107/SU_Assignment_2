import os
import numpy as np
from lid_features import extract_mfcc

def load_data(base_path="data/lid"):
    X = []
    y = []

    for label, folder in enumerate(["english", "hindi"]):
        path = os.path.join(base_path, folder)

        for file in os.listdir(path):
            if file.endswith(".wav"):
                file_path = os.path.join(path, file)

                features = extract_mfcc(file_path)

                X.append(features)
                y.append(label)

    X = np.array(X)
    y = np.array(y)

    return X, y
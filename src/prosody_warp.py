import parselmouth
from dtw import dtw
import numpy as np

ORIGINAL_AUDIO = "data/original_segment.wav"
GENERATED_AUDIO = "outputs/output_cloned.wav"

def extract_pitch(file):
    snd = parselmouth.Sound(file)
    pitch = snd.to_pitch()
    return pitch.selected_array['frequency']

def main():
    print("Extracting pitch...")

    f0_original = extract_pitch(ORIGINAL_AUDIO)
    f0_generated = extract_pitch(GENERATED_AUDIO)

    f0_original = f0_original[f0_original > 0]
    f0_generated = f0_generated[f0_generated > 0]

    print("Applying DTW...")

    alignment = dtw(f0_original, f0_generated, keep_internals=True)

    print("DTW distance:", alignment.distance)

if __name__ == "__main__":
    main()
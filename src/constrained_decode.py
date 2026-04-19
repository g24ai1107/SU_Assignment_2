from transformers import pipeline
import json

AUDIO_PATH = "data/clean.wav"
OUTPUT_PATH = "outputs/transcript_constrained.json"

# Important domain words (your syllabus terms)
BIAS_WORDS = [
    "stochastic",
    "cepstrum",
    "bayes",
    "likelihood",
    "probability",
    "distribution"
]


def apply_bias(text):
    words = text.split()

    corrected = []
    for word in words:
        for bias_word in BIAS_WORDS:
            if word.lower().startswith(bias_word[:4]):
                word = bias_word
        corrected.append(word)

    return " ".join(corrected)


def constrained_transcription():
    print("Loading Whisper...")

    pipe = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-small"
    )

    print("Transcribing...")
    result = pipe(AUDIO_PATH, return_timestamps=True)

    original_text = result["text"]
    corrected_text = apply_bias(original_text)

    result["original_text"] = original_text
    result["biased_text"] = corrected_text

    print("Saving...")
    with open(OUTPUT_PATH, "w") as f:
        json.dump(result, f, indent=4)

    print("\n--- BEFORE ---\n")
    print(original_text[:200])

    print("\n--- AFTER (BIAS APPLIED) ---\n")
    print(corrected_text[:200])


if __name__ == "__main__":
    constrained_transcription()
from transformers import pipeline
import json

AUDIO_PATH = "data/clean.wav"
OUTPUT_PATH = "outputs/transcript.json"

def transcribe_audio():
    print("Loading Whisper model...")
    
    pipe = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-small"
    )

    print("Transcribing audio...")
    result = pipe(AUDIO_PATH, return_timestamps=True)

    print("Saving transcript...")
    with open(OUTPUT_PATH, "w") as f:
        json.dump(result, f, indent=4)

    print("Done!")
    print("Sample Output:")
    print(result["text"][:200])  # preview first 200 chars

if __name__ == "__main__":
    transcribe_audio()
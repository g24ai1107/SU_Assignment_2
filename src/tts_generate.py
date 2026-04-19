from TTS.api import TTS

TEXT_PATH = "outputs/lrl_output.txt"
SPEAKER_WAV = "data/student_voice_ref.wav"
OUTPUT_PATH = "outputs/output_cloned.wav"

def main():
    print("Loading model...")

    tts = TTS(
        model_name="tts_models/multilingual/multi-dataset/your_tts",
        progress_bar=True,
        gpu=False
    )

    with open(TEXT_PATH, "r") as f:
        text = f.read()

    print("Generating speech...")

    tts.tts_to_file(
        text=text[:1000],  # limit initially
        speaker_wav=SPEAKER_WAV,
        language="en",
        file_path=OUTPUT_PATH
    )

    print("Done! Audio saved at:", OUTPUT_PATH)

if __name__ == "__main__":
    main()
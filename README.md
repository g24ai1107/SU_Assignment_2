# Speech Understanding Assignment 2

# Overview

This project implements an end-to-end pipeline for processing code-switched (Hinglish) speech.
The system transcribes lecture audio, identifies language switches, processes text, and generates synthesized speech using voice cloning.


# Features
Speech-to-Text using Whisper
Language Identification (Hindi vs English)
Constrained decoding with domain vocabulary
Audio denoising (spectral subtraction)
Hinglish to IPA conversion
Text normalization
Zero-shot voice cloning (YourTTS)
Prosody alignment using DTW
Anti-spoofing detection
Adversarial attack (FGSM)

---

## Project Structure

```
speech_assignment_2/
│
├── data/
│   ├── original_segment.wav
│   ├── clean.wav
│   ├── denoised.wav
│   └── student_voice_ref.wav
│
├── outputs/
│   ├── transcript.json
│   ├── transcript_constrained.json
│   ├── ipa_output.txt
│   ├── lrl_output.txt
│   └── output_cloned.wav
│
├── models/
│   └── lid_model.pth
│
├── src/
│   ├── preprocess.py
│   ├── transcribe.py
│   ├── constrained_decode.py
│   ├── denoise.py
│   ├── lid_model.py
│   ├── train_lid.py
│   ├── infer_lid.py
│   ├── ipa_convert.py
│   ├── translate_lrl.py
│   ├── tts_generate.py
│   ├── prosody_warp.py
│   ├── anti_spoof.py
│   └── fgsm_attack.py
│
├── README.md
└── requirements.txt
```

---


### 2. Install Dependencies

#### Base Environment

```bash
pip install torch torchaudio transformers librosa numpy scikit-learn
```

#### TTS Environment

python3.10 -m venv tts_env
source tts_env/bin/activate
pip install TTS


#### DTW Environment

python3.10 -m venv dtw_env
source dtw_env/bin/activate
pip install numpy scipy dtw-python praat-parselmouth librosa


---

## How to Run

### Preprocessing

python src/preprocess.py

### Transcription

python src/transcribe.py

### Constrained Decoding

python src/constrained_decode.py

### Denoising

python src/denoise.py

### LID Training

python src/train_lid.py

### LID Inference

python src/infer_lid.py

### IPA Conversion

python src/ipa_convert.py


### Text Processing

python src/translate_lrl.py


### Voice Cloning (TTS)

python src/tts_generate.py

### Prosody Alignment (DTW)

python src/prosody_warp.py


### Anti-Spoofing

python src/anti_spoof.py


### Adversarial Attack

python src/fgsm_attack.py


---

# Key Components
LID Model: BiLSTM using MFCC features
Denoising: Spectral subtraction
TTS: YourTTS (voice cloning)
Prosody: DTW-based alignment
---

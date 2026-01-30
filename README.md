# PII Audio Anonymization

A tool for detecting and anonymizing Personally Identifiable Information (PII) in audio files. It uses advanced speech-to-text models (WhisperX, NeMo, Qwen3-ASR) to transcribe audio and Named Entity Recognition (NER) models to detect sensitive information, which is then masked in the audio.

## Installation

This project requires Python 3.8+ and `ffmpeg` installed on your system.

### Dependencies

Install the required Python packages:

```bash
pip install whisperx nemo_toolkit[asr] librosa torch transformers pydub gradio soundfile numpy tqdm qwen-asr
```

**Note**: `whisperx` ,`nemo_toolkit`, and `qwen-asr` might have specific installation instructions depending on your CUDA version. Please refer to their official documentation.

### FFmpeg

Ensure `ffmpeg` is installed and available in your system path.

- **Ubuntu/Debian**: `sudo apt install ffmpeg`
- **MacOS**: `brew install ffmpeg`

## Usage

### Command Line Interface (CLI)

You can run the anonymization tool directly from the command line.

**Syntax:**

```bash
python pii_audio_anonimization.py <input_audio_file> <output_audio_file> [options]
```

**Arguments:**

- `audio_file`: Path to the input audio file (required).
- `output_file`: Path to the output audio file (required).
- `--framework`: Transcription framework to use. Options: `whisperx` (default), `nemo`, `qwen3asr`.
- `--model`: Transcription model name. Default: `large-v3`.
- `--anonymizer`: Anonymization method. Default: `eu-pii-safeguard`.
- `--align-model`: Alignment model (for WhisperX). Optional.
- `--token`: Hugging Face token (required for some models like `pyannote/speaker-diarization`).
- `--device`: Device to use. Default: `cuda`.
- `--language`: Language code. Default: `es` (Spanish).

**Examples:**

Using WhisperX:
```bash
python pii_audio_anonimization.py input.mp3 output.mp3 --framework whisperx --model large-v3
```

Using NeMo:
```bash
python pii_audio_anonimization.py input.mp3 output.mp3 --framework nemo --model nvidia/canary-1b-v2
```

Using Qwen3-ASR:
```bash
python pii_audio_anonimization.py input.mp3 output.mp3 --framework qwen3asr --model Qwen/Qwen3-ASR-1.7B
```

### Gradio Web Application

For a user-friendly graphical interface, run the Gradio application:

```bash
python app.py
```

This will launch a web interface where you can:
1. Upload an audio file.
2. Select the transcription framework (WhisperX or NeMo).
3. Choose the model (Dropdown available for NeMo).
4. Provide your Hugging Face token if needed.
5. Process and download the anonymized audio.

## Anonymization Methods

Currently, the tool supports:

- **eu-pii-safeguard**: Uses the `tabularisai/eu-pii-safeguard` model to detect PII entities such as names, locations, ID numbers, etc., in the transcription. These detected entities are then mapped back to the audio timestamps and replaced with a beep sound.

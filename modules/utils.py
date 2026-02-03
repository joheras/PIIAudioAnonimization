import librosa
import numpy as np
import soundfile as sf
from pydub import AudioSegment
from pydub.generators import Sine
from tqdm import tqdm

def to_16k_mono_audio(file: str) -> np.ndarray:
    """
    Converts audio to 16k mono and saves it as .wav.

    Args:
        file (str): Path to the input audio file.

    Returns:
        np.ndarray: The audio array in 16k mono.
    """
    audio_arr, sr = librosa.load(file, sr=None)

    if len(audio_arr.shape) > 1:
        audio_arr = librosa.to_mono(audio_arr)

    if sr != 16000:
        audio_arr = librosa.resample(audio_arr, orig_sr=sr, target_sr=16000)

    sf.write(file+".wav",audio_arr,16000)

def audio_anonymisation(audioFile,outputFile,segments):
    """
    Anonymizes audio by replacing segments with a beep.

    Args:
        audioFile (str): Input audio file.
        outputFile (str): Output audio file.
        segments (list): List of (start, end) tuples to beeping.
    """
    audio = AudioSegment.from_file(audioFile)
    output = audio
    for (s,e) in tqdm(segments):
        beep = Sine(1000).to_audio_segment(duration=int((e-s)*1000)).apply_gain(-3)
        output = (
            output[:int(s*1000)] +
            beep +
            output[int(e*1000):]
        )
    output.export(outputFile, format="mp3")

#!/usr/bin/env python
# coding: utf-8

import argparse
from modules import (
    TranscribeAndAlignNemo,
    TranscribeAndAlignWhisperX,
    TranscribeAndAlignQwen3ASR,
    EUPIISafeguard,
    Presidio,
    OpenMed,
    audio_anonymisation
)

def transcribe_and_align(audioFile,framework,model,**kwargs):
    """
    Factory function to transcribe and align audio using specified framework.

    Args:
        audioFile (str): Path to audio file.
        framework (str): 'whisperx', 'nemo', 'qwen3asr'.
        model (str): Model name.

    Returns:
        list: Transcription segments.
    """
    if framework == "nemo":
        transcriptor = TranscribeAndAlignNemo(audioFile,model,**kwargs)
    elif framework == "whisperx":
        transcriptor = TranscribeAndAlignWhisperX(audioFile,model,**kwargs)
    elif framework == "qwen3asr":
        transcriptor = TranscribeAndAlignQwen3ASR(audioFile,**kwargs)
    else:
        raise Exception("Framework not supported")
        
    return transcriptor.transcribe_and_align()

def entities_to_anonimize(transcription,anonymizerMethod="eu-pii-safeguard",pii_types_to_detect=None,**kwargs):
    """
    Detects entities to anonymize using the specified method.

    Args:
        transcription (list): Transcription segments.
        anonymizerMethod (str): Method to use (default: "eu-pii-safeguard").
        pii_types_to_detect (list): List of PII types.

    Returns:
       list: List of intervals to anonymize.
    """
    if(anonymizerMethod=='eu-pii-safeguard'):
        anonymizer = EUPIISafeguard(transcription,pii_types_to_detect,**kwargs)
    elif(anonymizerMethod=='presidio'):
        anonymizer = Presidio(transcription,pii_types_to_detect,**kwargs)
    elif(anonymizerMethod=='openmed'):
        anonymizer = OpenMed(transcription,pii_types_to_detect,**kwargs)
    else:
        raise Exception("Not supported method")
    
    return anonymizer.anonymise()

def pii_audio_anonimization(audioFile,outputFile,transcriptionFramework,transcriptionModel,anonymizerMethod,alignModel=None,**kwargs):
    """
    Main function to perform PII audio anonymization.

    Args:
        audioFile (str): Input audio file path.
        outputFile (str): Output audio file path.
        transcriptionFramework (str): 'whisperx' or 'nemo'.
        transcriptionModel (str): Transcription model name.
        anonymizerMethod (str): Anonymization method.
        alignModel (str, optional): Alignment model for WhisperX.
        **kwargs: Additional arguments for models.
    """
    print(["Transcribing and aligning audio..."])
    transcription = transcribe_and_align(audioFile,transcriptionFramework,transcriptionModel,**dict(kwargs,alignModel=alignModel))
    print(["Detecting PII..."])
    segments = entities_to_anonimize(transcription,anonymizerMethod,**kwargs)
    print(["Audio PII anonimisation..."])
    audio_anonymisation(audioFile,outputFile,segments)
    print(["Done!"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PII Audio Anonymization CLI")
    parser.add_argument("audio_file", type=str, help="Path to the input audio file")
    parser.add_argument("output_file", type=str, help="Path to the output audio file")
    parser.add_argument("--framework", type=str, default="whisperx", help="Transcription framework (default: whisperx)")
    parser.add_argument("--model", type=str, default="large-v3", help="Transcription model (default: large-v3)")
    parser.add_argument("--anonymizer", type=str, default="eu-pii-safeguard", help="Anonymizer method (default: eu-pii-safeguard)")
    parser.add_argument("--align-model", type=str, default=None, help="Alignment model (optional)")
    parser.add_argument("--token", type=str, default=None, help="Hugging Face token (optional)")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (default: cuda)")
    parser.add_argument("--language", type=str, default="es", help="Language code (default: es)")

    args = parser.parse_args()

    pii_audio_anonimization(
        audioFile=args.audio_file,
        outputFile=args.output_file,
        transcriptionFramework=args.framework,
        transcriptionModel=args.model,
        anonymizerMethod=args.anonymizer,
        alignModel=args.align_model,
        token=args.token,
        device=args.device,
        language=args.language
    )

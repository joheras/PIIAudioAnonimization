#!/usr/bin/env python
# coding: utf-8

# In[1]:


import nemo.collections.asr as nemo_asr
from nemo.collections.asr.models import ASRModel
from qwen_asr import Qwen3ASRModel
import librosa
import whisperx
import gc
from whisperx.diarize import DiarizationPipeline
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
import string
from pydub import AudioSegment
from pydub.generators import Sine
import re
import string
from tqdm import tqdm
import numpy as np
import soundfile as sf
import argparse


# In[2]:


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

class TranscribeAndAlign():
    """
    Base class for transcription and alignment.
    """
    
    def __init__(self,audioFile):
        self.audioFile = audioFile
        
    def transcribe_and_align():
        """
        Abstract method for transcription and alignment.
        """
        pass

class TranscribeAndAlignWhisperX(TranscribeAndAlign):
    """
    Transcription and alignment using WhisperX.
    """
    
    def __init__(self,audioFile,model="large-v3",device="cuda",compute_type="float16",device_index=2,language="es",batch_size=16,align_model=None,**kwargs):
        """
        Initializes the WhisperX transcriber.

        Args:
           audioFile (str): Path to audio file.
           model (str): WhisperX model name.
           device (str): Device to use (cuda/cpu).
           compute_type (str): Compute type (float16/float32).
           device_index (int): Device index.
           language (str): Language code.
           batch_size (int): Batch size.
           align_model (str): Alignment model name.
        """
        super().__init__(audioFile)
        self.model=model
        self.device=device
        self.compute_type = compute_type
        self.device_index=device_index
        self.language=language
        self.batch_size=batch_size
        self.align_model=align_model
    
        
    def transcribe_and_align(self):
        """
        Transcribes and aligns the audio using WhisperX.

        Returns:
            list: List of segments with word timestamps.
        """
        modelW = whisperx.load_model(self.model,self.device,device_index=self.device_index,language=self.language,compute_type=self.compute_type)
        audio = whisperx.load_audio(self.audioFile)
        result = modelW.transcribe(audio, batch_size=self.batch_size)
        if self.align_model is not None:
            model_a, metadata = whisperx.load_align_model(self.align_model,language_code=self.language, device=self.device)
        else:
            model_a, metadata = whisperx.load_align_model(language_code=self.language, device=self.device)
        result_a = whisperx.align(result["segments"], model_a, metadata, audio,self.device, return_char_alignments=False)
        del modelW
        del model_a
        return result_a["segments"]
    
    
class TranscribeAndAlignNemo(TranscribeAndAlign):
    """
    Transcription and alignment using NeMo.
    """
    
    def __init__(self,audioFile,model="nvidia/canary-1b-v2",language="es",map_location="cuda",**kwargs):
        """
        Initializes the NeMo transcriber.

        Args:
            audioFile (str): Path to audio file.
            model (str): NeMo model name.
            language (str): Language code.
            map_location (str): Device map location.
        """
        super().__init__(audioFile)
        self.model=model
        self.map_location=map_location
        self.language=language
        
    def transcribe_and_align(self):
        """
        Transcribes and aligns the audio using NeMo.

        Returns:
            list: List of segments with word timestamps.
        """
        
        to_16k_mono_audio(self.audioFile)
        
        
        if 'canary' in self.model:
            asr_model = ASRModel.from_pretrained(model_name=self.model,map_location=self.map_location)
            output = asr_model.transcribe([self.audioFile+".wav"], batch_size=4,timestamps=True,source_lang=self.language, target_lang=self.language)
        else:
            asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name=self.model,map_location="cuda")
            output = asr_model.transcribe([self.audioFile+".wav"], timestamps=True)
            
        word_timestamps = output[0].timestamp['word'] # word level timestamps for first sample
        segment_timestamps = output[0].timestamp['segment'] # segment level timestamps
        for s in segment_timestamps:
            s['text']=s['segment']
            del s['segment']

        # Combining segments and words 
        i = 0
        j = 0
        while(i<len(segment_timestamps)):    
            while(j<len(word_timestamps)):
                # print(str(i)+" out of " + str(len(segment_timestamps)))
                if(word_timestamps[j]['end']<=segment_timestamps[i]['end']):
                    if('words' in segment_timestamps[i]):
                        segment_timestamps[i]['words'].append(word_timestamps[j])
                    else:
                        segment_timestamps[i]['words'] = [word_timestamps[j]]
                    j+=1
                else:
                    i+=1
            i+=1
            
        return segment_timestamps

class TranscribeAndAlignQwen3ASR(TranscribeAndAlign):
    """
    Transcription and alignment using Qwen3-ASR.
    """
    
    def __init__(self,audioFile,model="Qwen/Qwen3-ASR-1.7B",aligner="Qwen/Qwen3-ForcedAligner-0.6B",device_map="cuda:0",language="Spanish",batch_size=32,max_new_tokens=8092,**kwargs):
        """
        Initializes the Qwen3-ASR transcriber.

        Args:
           audioFile (str): Path to audio file.
           device_map (str): Device map.
           language (str): Language code.
           batch_size (int): Batch size.
           max_new_tokens (int): Maximum number of new tokens.
        """
        super().__init__(audioFile)
        self.device_map=device_map
        self.language=language
        self.batch_size=batch_size
        self.max_new_tokens=max_new_tokens
        self.model=model
        self.aligner=aligner
        
    def transcribe_and_align(self):
        """
        Transcribes and aligns the audio using WhisperX.

        Returns:
            list: List of segments with word timestamps.
        """
        model = Qwen3ASRModel.from_pretrained(
            self.model,
            dtype=torch.bfloat16,
            device_map=self.device_map,
            # attn_implementation="flash_attention_2",
            max_inference_batch_size=self.batch_size, # Batch size limit for inference. -1 means unlimited. Smaller values can help avoid OOM.
            max_new_tokens=self.max_new_tokens, # Maximum number of tokens to generate. Set a larger value for long audio input.
            forced_aligner=self.aligner,
            forced_aligner_kwargs=dict(
                dtype=torch.bfloat16,
                device_map=self.device_map,
                # attn_implementation="flash_attention_2",
            ),
        )

        results = model.transcribe(
            audio=[
            self.audioFile,
            ],
            language=[self.language], # can also be set to None for automatic language detection
            return_time_stamps=True,
        )
        del model
        segment_timestamps = [{'segment':results[0].text,'words':[{'word':w.text,'start':w.start_time,'end':w.end_time} for w in results[0].time_stamps]}]
        return segment_timestamps


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
          



# transcription = transcribe_and_align("../PersonaBeneficiaria04diciembre2025.m4a","whisperx","large-v3")
# transcribe_and_align("../PersonaBeneficiaria04diciembre2025.m4a","nemo","nvidia/canary-1b-v2")
# transcribe_and_align("../PersonaBeneficiaria04diciembre2025.m4a","nemo","nvidia/parakeet-tdt-0.6b-v3")


# In[3]:


class Anonymizer():
    """
    Base class for anonymization.
    """
    
    def __init__(self,transcription):
        self.transcription=transcription
        

class EUPIISafeguard(Anonymizer):
    """
    Anonymizer using EU-PII-Safeguard model.
    """
    
    
    
    def __init__(self,transcription,types,token,**kwargs):
        """
        Initializes the EUPIISafeguard anonymizer.

        Args:
            transcription (list): Transcription segments.
            types (list): List of PII types to detect.
            token (str): Hugging Face token.
        """
        super().__init__(transcription)
        self.PII_TYPES_TO_DETECT = ['ACCOUNT_NUMBER', 'ADDRESS', 'AGE', 'AMOUNT', 'BUILDING_NUMBER',
       'CITY', 'COMPANY_NAME', 'COUNTRY', 'CREDIT_CARD_NUMBER',
       'CURRENCY', 'DEVICE_ID', 'DOB', 'DRIVER_LICENSE', 'EMAIL',
       'ETHNICITY', 'FIRSTNAME', 'GENDER', 'HEALTH_CONDITION',
       'HEALTH_INSURANCE_ID', 'IBAN', 'IP_ADDRESS', 'JOB_TITLE',
       'LASTNAME', 'LATITUDE', 'LONGITUDE', 'MAC_ADDRESS', 'MIDDLENAME',
       'NATIONAL_ID', 'PASSPORT_NUMBER', 'PASSWORD', 'PHONE_NUMBER',
       'POLITICAL_OPINION', 'POSTAL_CODE', 'PREFIX', 'RELIGION', 'SALARY',
       'SEXUAL_ORIENTATION', 'STATE', 'STREET', 'TAX_ID', 'URL',
       'USERNAME']
        
        self.types = self.PII_TYPES_TO_DETECT if types is None else types
        self.token = token
        self.model_name = "tabularisai/eu-pii-safeguard"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name,token=token)
        self.model = AutoModelForTokenClassification.from_pretrained(self.model_name,token=token)
            
    def anonymise(self):
        """
        Detects PII in the transcription.

        Returns:
            list: List of tuples (start, end) for PII segments.
        """
        translator = str.maketrans('', '', string.punctuation)
        toAnonymise = []

        for res in self.transcription:
            text = res['text']
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True)
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.argmax(outputs.logits, dim=-1)

            # Get predictions
            tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
            predicted_labels = [self.model.config.id2label[pred.item()] for pred in predictions[0]]
            # print("Detected PII:")
            detectedWords = []
            detectedLabels = []
            temp=""
            tempLabel=""
            for token, label in zip(tokens, predicted_labels):
                if label != "O":
                    if(label.split('-')[1] in self.types):
                        if(token[0]=="▁"):
                            if(temp!=""):
                                # print(f"  {tempLabel}: {temp}")
                                detectedWords.append(temp)
                                detectedLabels.append(tempLabel)
                            temp=token[1:]
                            tempLabel=label
                        else:
                            temp+=token

            if(temp!=""):
                detectedWords.append(temp)
                detectedLabels.append(tempLabel)
            if(len(detectedWords)>0):
                i=0
                # print(list(zip(detectedWords,detectedLabels)))
                for word in res['words']:
                    if(word['word'].translate(translator)==detectedWords[i]):
                        i+=1
                        toAnonymise.append((word['start'],word['end']))
                        if(len(detectedWords)==i):
                            break
        
        return toAnonymise
        

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
    else:
        raise Exception("Not supported method")
    
    return anonymizer.anonymise()
        

    
# entities_to_anonimize(transcription,token="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")


# In[4]:


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


# In[5]:


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

        
#pii_audio_anonimization("../PersonaBeneficiaria04diciembre2025.m4a","output.mp3","whisperx","large-v3","eu-pii-safeguard",token="XXX")


# In[ ]:





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

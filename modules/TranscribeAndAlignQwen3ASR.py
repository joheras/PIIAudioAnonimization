from qwen_asr import Qwen3ASRModel
import torch
from .TranscribeAndAlign import TranscribeAndAlign

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

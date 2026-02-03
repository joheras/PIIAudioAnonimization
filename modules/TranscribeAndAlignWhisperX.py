import whisperx
from .TranscribeAndAlign import TranscribeAndAlign

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

import nemo.collections.asr as nemo_asr
from nemo.collections.asr.models import ASRModel
from .TranscribeAndAlign import TranscribeAndAlign
from .utils import to_16k_mono_audio

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

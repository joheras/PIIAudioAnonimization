from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
from presidio_analyzer.predefined_recognizers import EmailRecognizer
from presidio_analyzer.nlp_engine import NlpEngineProviderimport torch
import string
from .Anonymizer import Anonymizer


configuration = {
    "nlp_engine_name": "spacy",
    "models": [
        {"lang_code": "es", "model_name": "es_core_news_md"},
        {"lang_code": "en", "model_name": "en_core_web_lg"},
    ],
}

# Create NLP engine based on configuration
provider = NlpEngineProvider(nlp_configuration=configuration)
nlp_engine_with_spanish = provider.create_engine()

# Pass the created NLP engine and supported_languages to the AnalyzerEngine
analyzer = AnalyzerEngine(
    nlp_engine=nlp_engine_with_spanish, supported_languages=["en", "es"]
)

class Presidio(Anonymizer):
    """
    Anonymizer using Presidio model.
    """
    
    
    
    def __init__(self,transcription,types,language='es',**kwargs):
        """
        Initializes the Presidio anonymizer.

        Args:
            transcription (list): Transcription segments.
            types (list): List of PII types to detect.
        """
        super().__init__(transcription)
        self.PII_TYPES_TO_DETECT = ['CREDIT_CARD','CRYPTO','DATE_TIME','EMAIL_ADDRESS','IBAN_CODE','NRP','LOCATION','PERSON',
                      'PHONE_NUMBER','ES_NIF','ES_NIE']
        
        self.types = self.PII_TYPES_TO_DETECT if types is None else types
        self.language=language
        
            
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
            results = analyzer.analyze(text=text,language=self.language)
            predicted_labels = [x.entity_type for x in results]
            tokens = [text[x.start:x.end] for x in results]
            # print("Detected PII:")
            detectedWords = []
            detectedLabels = []
            temp=""
            tempLabel=""
            for token, label in zip(tokens, predicted_labels):
                if label in self.types:
                    detectedWords.append(token)
                    detectedLabels.append(label)
            if(len(detectedWords)>0):
                i=0
                for word in res['words']:
                    if(word['word'].translate(translator)==detectedWords[i]):
                        i+=1
                        toAnonymise.append((word['start'],word['end']))
                        if(len(detectedWords)==i):
                            break
        
        return toAnonymise

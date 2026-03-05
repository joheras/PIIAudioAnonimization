from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
import string
from .Anonymizer import Anonymizer

class OpenMed(Anonymizer):
    """
    Anonymizer using OpenMed models.
    """
    
    
    
    def __init__(self,transcription,types,token,model_name="OpenMed/OpenMed-PII-Spanish-SnowflakeMed-Large-568M-v1",**kwargs):
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
        self.model_name = model_name
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

import unicodedata
import langdetect
from langdetect import DetectorFactory

class PreprocessingService:
    """
        Preprocessing Service: A class that help preprocessing input from users, make sure reducing costs
    """
    
    @staticmethod
    def remove_whitespace(string: str) -> str:    
        string = string.replace("\n\n", "").replace("\n", "")
        string = string.strip()

        return string

    @staticmethod
    def normalize(string: str) -> str:
        return unicodedata.normalize("NFC", string)


    @staticmethod
    def detect_language(string: str, seed: int = 0) -> str:
        DetectorFactory.seed = seed
        return langdetect.detect(string)

    


import unicodedata
import langdetect
from langdetect import DetectorFactory
from typing import List, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.services.base import BaseService


class PreprocessingService(BaseService):
    """
        Preprocessing Service: A class that help preprocessing input from users, make sure reducing costs
    """
    
    @staticmethod
    def _remove_whitespace(string: str) -> str:    
        string = string.replace("\n\n", "").replace("\n", "")
        string = string.strip()
        string = " ".join(string.split())

        return string

    @staticmethod
    def _normalize(string: str) -> str:
        return unicodedata.normalize("NFC", string)

    @classmethod
    def preprocess(cls, string: str) -> str:
        string = cls.remove_whitespace(string)
        string = cls.normalize(string)

        return string

    @staticmethod
    def detect_language(string: str, seed: int = 0) -> str:
        DetectorFactory.seed = seed
        return langdetect.detect(string)

    @staticmethod
    def chunk(
        documents: List[str],
        seperator: Optional[List[str]] = None,

    ) -> List[str]:
        splitter = RecursiveCharacterTextSplitter()


    


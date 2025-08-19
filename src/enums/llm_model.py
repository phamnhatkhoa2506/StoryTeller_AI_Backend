from enum import Enum


class LLMModelEnum(str, Enum):
    GEMINI = "gemini"
    JINA = "jina"
    GROQ = "groq"
    HUGGINGFACE = "hf"
    COHERE = "cohere"
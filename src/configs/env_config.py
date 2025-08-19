from typing import List, Dict
from pydantic_settings import BaseSettings, SettingsConfigDict


class EnvConfig(BaseSettings):
    # Apps
    HOST: str = ""
    PORT: str = ""
    APP_URL: str = ""

    # URLs
    JINA_API_URL: str = ""

    # API Keys
    GOOGLE_API_KEY: str = ""
    COHERE_API_KEY: str = ""
    GROQ_API_KEY: str = ""
    JINA_API_KEY: str = ""
    HUGGINGFACE_API_KEY: str = ""

    # LLMs
    JINA_CLASSIFICATION_MODEL: str = 'jina-embeddings-v3'
    JINA_EMBEDDING_MODEL: str='jina-embeddings-v3'
    JINA_RERANKING_MODEL: str='jina-reranker-v2-base-multilingual'
    COHERE_CHAT_MODEL: str='command-a-03-2025'
    COHERE_CLASSIFICATION_MODEL: str=''
    COHERE_EMBEDDING_MODEL: str='embed-v4.0'
    COHERE_RERANKING_MODEL: str='rerank-v3.5'
    GROQ_CHAT_MODEL: str = 'llama-3.1-8b-instant'
    HUGGINGFACE_CHAT_MODEL: str ='mistralai/Mistral-7B-Instruct-v0.2'
    HUGGINGFACE_CLASSIFICATION_MODEL: str = ''
    HUGGINGFACE_EMBEDDING_MODEL: str = 'sentence-transformers/all-mpnet-base-v2'
    HUGGINGFACE_RERANKING_MODEL: str = ''
    GEMINI_EMBEDDING_MODEL: str = 'models/gemini-embedding-001'

    # Image Generation Model
    IMAGEN_4_MODEL: str = 'imagen-4.0-generate-001'
    GEMINI_SPEECH_MODEL: str = 'gemini-2.5-flash-preview-tts'


    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    @staticmethod
    def get_list_models() -> List[Dict[str, str]]:
        pass


env_config = EnvConfig()
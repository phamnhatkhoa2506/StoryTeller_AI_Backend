from ast import List
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

from src.api import JinaAPI, CohereAPI
from src.enums import LLMModelEnum
from src.logger.logger import Logger
from src.configs import env_config


logger = Logger()


class GeminiEmbeddingModel(object):
    def __init__(self, model: str):
        self.model = GoogleGenerativeAIEmbeddings(
            model='models/gemini-embedding-001',
            google_api_key=env_config.GOOGLE_API_KEY
        )

    def embed(self, documents: list[str], task: str) -> List[List[float]]:
        return self.model.embed_documents(documents, task_type=task)


class HuggingFaceEmbeddingModel(object):
    def __init__(self, model: str):
        pass


class EmbeddingModel(object):
    """
        Embedding model
    """

    def __init__(
        self, 
        model: str,
        framework: str,
        name: str,
        **kwargs
    ) -> None:
        
        self.name = name
        
        if LLMModelEnum.GEMINI == framework:
            self.model = GoogleGenerativeAIEmbeddings(
                model=model,
                google_api_key=env_config.GOOGLE_API_KEY
            )
        elif LLMModelEnum.HUGGINGFACE  == framework:
            model_kwargs = kwargs.get("model_kwargs", {})
            encode_kwargs = kwargs.get("encode_kwargs", {})

            self.model = HuggingFaceEmbeddings(
                model_name=model,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs
            )
        elif LLMModelEnum.COHERE  == framework:
            self.model = CohereAPI()
        elif LLMModelEnum.JINA  == framework:
            self.model = JinaAPI()
        else:
            raise ValueError("Model not supported")

        logger.info(f"Load chat model {model} succesfully")

    def embed(self, **kwargs) -> str:
        """
            The embedding method
        """

        return self.model.embed(**kwargs)

    def __call__(self, **kwargs) -> str:
        return self.embed(**kwargs)
from typing import Any, List, Optional
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

from src.api import JinaAPI, CohereAPI
from src.enums import LLMModelEnum
from src.logger.logger import Logger
from src.configs import env_config
from src.ml_models.base import BaseMLModel, BaseEmbeddingModel


logger = Logger()


class GeminiEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, model: str):
        self.model = GoogleGenerativeAIEmbeddings(
            model='models/gemini-embedding-001',
            google_api_key=env_config.GOOGLE_API_KEY
        )

    def __embed(self, documents: list[str], task: Optional[str] = None) -> List[List[float]]:
        return self.model.embed_documents(documents, task_type=task)

    def __call__(
        self, 
        documents: list[str], 
        task: Optional[str] = None,
        **kwargs,
    ) -> List[List[float]]:
        return self.__embed(documents=documents, task=task)


class HuggingFaceEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, model: str):
        pass

    def __call__(
        self, 
        documents: list[str], 
        task: Optional[str] = None,
        **kwargs,
    ) -> List[List[float]]:
        pass


class CohereEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, model: str):
        self.api = CohereAPI()
        self.model = model

    def __embed(self, documents: list[str], task: Optional[str] = None) -> List[List[float]]:

        inputs = [{
            "content": [
                {"type": "text", "text": doc} for doc in documents
            ],
        },]

        return self.api.embed(
            model=self.model,
            input=inputs,
            task=task,
        )

    def __call__(
        self, 
        documents: list[str], 
        task: Optional[str] = None,
        **kwargs,
    ) -> List[List[float]]:
        return self.__embed(documents=documents, task=task)


class JinaEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, model: str):
        self.api = JinaAPI()
        self.model = model

    def __embed(self, documents: list[str], task: Optional[str] = None) -> List[List[float]]:

        return self.api.embed(
            model=self.model,
            input=documents,
            task=task,
        )

    def __call__(
        self, 
        documents: list[str], 
        task: Optional[str] = None,
        **kwargs,
    ) -> List[List[float]]:
        return self.__embed(documents=documents, task=task)


class EmbeddingModel(BaseMLModel):
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
        
        match framework:
            case LLMModelEnum.GEMINI:
                self.model = GeminiEmbeddingModel(
                    model=model,
                )

            case LLMModelEnum.HUGGINGFACE:
                model_kwargs = kwargs.get("model_kwargs", {})
                encode_kwargs = kwargs.get("encode_kwargs", {})

                self.model = HuggingFaceEmbeddings(
                    model_name=model,
                    model_kwargs=model_kwargs,
                    encode_kwargs=encode_kwargs
                )

            case LLMModelEnum.COHERE:
                self.model = CohereEmbeddingModel(model=model)
                
            case LLMModelEnum.JINA:
                self.model = JinaEmbeddingModel(model=model)

            case _:
                raise ValueError("Model not supported")

        logger.info(f"Load chat model {model} succesfully")

    def embed_documents(self, documents: list[str], task: str = "classification") -> List[List[float]]:
        """ Method for embedding function in vector database """

        try:
            return self.model(
                documents=documents,
                task=task
            )
        except Exception as e:
            logger.info(f"Error: {str(e)}")

            return []

    def __call__(self, *args: Any, **kwargs: Any) -> List[List[float]]:

        return self.model(**kwargs)
from src.api import JinaAPI, CohereAPI
from src.enums import LLMModelEnum
from src.logger.logger import Logger
from src.configs import env_config
from src.ml_models.base import BaseRerankingModel, BaseMLModel


logger = Logger()


class RerankingModel(BaseMLModel):
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
        
        if LLMModelEnum.COHERE  == framework:
            self.model = CohereAPI()
        elif LLMModelEnum.JINA  == framework:
            self.model = JinaAPI()
        else:
            raise ValueError("Model not supported")

        logger.info(f"Load chat model {model} succesfully")

    def rerank(self, **kwargs) -> str:
        """
            The reranking method
        """

        return self.model.rerank(**kwargs)

    def __call__(self, **kwargs) -> str:
        return self.rerank(**kwargs)
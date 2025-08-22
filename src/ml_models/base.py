from abc import ABC, abstractmethod
from typing import List, Optional, Any


class BaseMLModel(ABC):
    
    @abstractmethod
    def __call__(self, *args: Any, **kwargs) -> Any:
        pass


class BaseChatModel(ABC):
    @abstractmethod
    def __call__(self, input: Any, **kwargs) -> str:
        pass


class BaseClassificationModel(ABC):
    @abstractmethod
    def __call__(self, **params) -> str | List[str]:
        pass


class BaseEmbeddingModel(ABC):
    """ Base Class for embedding model """

    @abstractmethod
    def __call__(
        self, 
        documents: list[str], 
        task: Optional[str] = None,
        **kwargs
    ) -> List[List[float]]:
        pass


class BaseRerankingModel(ABC):
    pass
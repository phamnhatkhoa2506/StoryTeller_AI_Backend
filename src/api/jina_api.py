import requests
from typing import List, Dict, Optional

from src.configs import env_config
from src.logger.logger import Logger
from src.api.base import BaseAPI
from src.configs import env_config


logger = Logger()


class JinaAPI(BaseAPI):
    """API Class calling APIs from Jina Server"""

    def __init__(self) -> None:
        self.urls = {
            "classifier": f"{env_config.JINA_API_URL}/classify",
            "embedding" : f"{env_config.JINA_API_URL}/embeddings",
            "reranking" : f"{env_config.JINA_API_URL}/rerank",
         }

    def classify(
        self, 
        model: str, 
        input: List[str], 
        labels: List[str],
        **kwargs
    ) -> Dict:
        """
            Call the classification API

            Parameters:
                - model: model name for cohere api
                - input: list of strings of input texts
                - labels: list of labels

            Return the result from api with example format
            {
                "object": "classification",
                "index": 0,
                "prediction": "Simple task",
                "score": 0.3522600736981647,
                "predictions": [
                    {
                        "label": "Simple task",
                        "score": 0.3522600736981647
                    },
                    {
                        "label": "Complex reasoning",
                        "score": 0.3413496497255639
                    },
                    {
                        "label": "Creative writing",
                        "score": 0.3063902765762714
                    }
                ]
            },
        """

        if not labels or not input:
            return

        data = {
            "model": model,
            "input": input,
            "labels": labels,
            **kwargs
        }

        try:
            response = requests.post(
                self.urls["classifier"],
                headers={
                    "authorization": f"Bearer {env_config.JINA_API_KEY}",
                    "content-type": "application/json"
                },
                json=data
            )

            if response.status_code == 200:
                response = response.json()

                labels = [
                    obj["prediction"] for obj in response["data"]
                ]

                logger.info(f"Classfier: labels {labels}")

                return labels

            else:
                logger.error(f"Call classification API failed, code: {response.status_code}, response: {response.json()}")

        except Exception as e:
            return {}

    def embed(
        self,  
        model: str, 
        input: List[str], 
        task: Optional[str] = None,
        **kwargs
    ) -> List[List[float]]:
        """
            Call the embedding API

            Parameters:
                - model: model name for cohere api
                - input: list of strings of input texts with for mat
                - task: task for embedding include "text-matching", "classification", "seperation", ...
                - embedding_types: list of embedding type

            Return result from api
        """

        try:
            data = {
                "model": model,
                "input": input,
                "task": task,
                **kwargs
            }

            response = requests.post(
                self.urls["embedding"],
                headers={
                    "authorization": f"Bearer {env_config.JINA_API_KEY}",
                    "content-type": "application/json"
                },
                json=data
            )

            if response.status_code == 200:
                response = response.json()

                logger.info(f"Embedding: tokens {response['usage']}")

                response = [
                    obj["embedding"] for obj in response["data"]
                ]

                return response

            else:
                logger.error(f"Call embedding API failed, code: {response.status_code}, response: {response.json()}")

                return []
        except Exception as e:
            logger.error(f"Call embedding API failed: {str(e)}")

            return []

    def rerank(
        self,
        model: str,
        query: str,
        top_n: int,
        documents: List[str],
        **kwargs
    ) -> Dict | None:
        """
            Call the rerank API

            Call the reranking API

            Parameters:
                - model: model name for cohere api
                - input: list of strings of input texts with for mat
                - task: task for embedding include "text-matching", "classification", "seperation", ...
                - embedding_types: list of embedding type

            Return result from api with format below
            Examples:
            {
                "results": [
                    {
                        "index": 3,
                        "relevance_score": 0.999071
                    },
                    {
                        "index": 4,
                        "relevance_score": 0.7867867
                    },
                    {
                        "index": 0,
                        "relevance_score": 0.32713068
                    }
                ],
            }
        """

        if not query:
            return 

        if top_n > len(documents):
            raise ValueError("Top_n can not be bigger than length of documents")

        data = {
            "model": model,
            "query": query,
            "top_n": top_n,
            "documents": documents,
            **kwargs
        }

        response = requests.post(
            self.urls["reranking"],
            headers={
                "authorization": f"Bearer {env_config.JINA_API_KEY}",
                "content-type": "application/json"
            },
            json=data
        )

        if response.status_code == 200:
            response = response.json()

            logger.info(f"Rerank: tokens {response['usage']}")

            return {"results": response["results"]}

        else:
            logger.error(f"Call embedding API failed, code: {response.status_code}, response: {response.json()}")

            return None

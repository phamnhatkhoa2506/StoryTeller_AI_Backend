import json
import cohere
from cohere import ClassifyExample
from typing import Any, List, Dict, Optional
from typing_extensions import deprecated

from src.logger.logger import Logger
from src.configs import env_config
from src.api.base import BaseAPI


logger = Logger()


class CohereAPI(BaseAPI):
    """A class that call the api from Cohere"""

    def __init__(self) -> None:
        self.client = cohere.Client(api_key=env_config.COHERE_API_KEY)
        self.clientv2 = cohere.ClientV2(api_key=env_config.COHERE_API_KEY)

    @deprecated("The method can not be used because there's no finetuned models")
    def classify(
        self, 
        model: str, 
        input: List[str],
        labels:  List[Dict[str, str]],
        **kwargs
    ) -> Dict | None:
        """
            Call the classification API

            Parameters:
                - model: model name for cohere api
                - input: list of strings of input texts
                - labels: has format of    
                    [
                        {
                            "text": ...
                            "label": ...
                        },
                        ...
                    ]
                    , contains example text and label

            Return the result from api with format
            {
                "id": ...,
                "predictions": [
                    ...
                ],
                "confidence": [
                    ...
                ],
                "labels: {
                    "...": {
                        "confidence": ...
                    },
                    ...
                }
                "classification_type": ...,
                "input": ...,
                "prediction": ...,
                "confidence": ...
            }

        """

        if not labels or not input:
            return

        examples = [
            ClassifyExample(
                text=label["text"],
                label=label["label"]
            )
            for label in labels
        ]

        try:
            response = self.client.classify(
                model=model,
                inputs=input,
                examples=examples,
                **kwargs
            )

            if response.status_code == 200:
                response = response.json()

                logger.info(f"Classfier: {response['classifications'][0]}")

                return response["classifications"][0]

            else:
                logger.error(f"Call classification API failed, code: {response.status_code}, response: {response.json()}")

            return {}
        
        except Exception as e:
            logger.error(f"Call classification API failed: {str(e)}" )
            
            return {}

    def embed(
        self,  
        model: str, 
        input: List[Dict[str, Any]], 
        task: Optional[str] = None,
        embedding_types: List[str] = ["float"],
        **kwargs
    ) -> List[List[float]]:
        """
            Call the embedding API

            Parameters:
                - model: model name for cohere api
                - input: list of strings of input texts with format
                    [
                        {
                            "content": [
                                {"type": "text", "text": "hello"},
                                {"type": "text", "text": "goodbye"}
                            ]
                        },
                    ]
                - task: task for embedding include "text-matching", "classification", "seperation", ...
                - embedding_types: list of embedding type

            Return result from api with list of list of float
        """

        if not input:
            return 

        try:
            logger.info(f"Input: {input}")

            response = self.clientv2.embed(
                model=model,
                inputs=input,
                input_type=task,
                embedding_types=embedding_types,
                **kwargs
            )

            response = json.loads(response.json())

            logger.info(f"Json: {response}")

            return response["embeddings"]["float_"]
        
        except Exception as e:
            logger.error(f"Call embdedding API failed: {str(e)}" )
            
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

        try:
            response = self.clientv2.embed(
                model=model,
                query=query,
                documents=documents,
                top_n=top_n,
                **kwargs
            )

            if response.status_code == 200:
                response = response.json()

                logger.info(f"Reranked: {response['results']}")

                return {"results": response["results"]}

            else:
                logger.error(f"Call reranking API failed, code: {response.status_code}, response: {response.json()}")

            return {}
        
        except Exception as e:
            logger.error(f"Call reranking API failed: {str(e)}" )
            
            return {}


        

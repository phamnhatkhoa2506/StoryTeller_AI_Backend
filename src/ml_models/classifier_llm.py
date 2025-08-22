from typing import List, Any
from typing_extensions import deprecated
from langchain_core.messages import BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from src.api.jina_api import JinaAPI
from src.enums import LLMModelEnum, UserIntentEnum
from src.logger.logger import Logger
from src.configs import env_config
from src.api import CohereAPI, JinaAPI
from src.ml_models.base import BaseMLModel, BaseClassificationModel


logger = Logger()


class GeminiClassificationModel(BaseClassificationModel):
    """
        Classification Model Class for Gemini 
    """

    def __init__(
        self,
        model: str,
        prompt_template: List[BaseMessage]
    ) -> None:
        self.model = ChatGoogleGenerativeAI(
            model=model,
            temperature=0.0,
            max_tokens=1000,
            api_key=env_config.GOOGLE_API_KEY
        )
        self.prompt_template = prompt_template

    def classify(
        self, 
        **params
    ) -> str:
        """ 
            The method for classification

            Parameters:
                Require parameters in the prompt template

            Return the label
        """
        prompt_value = self.prompt_template.invoke({"requirement": params.get("requirement")})
       
        # If prompt_value has to_messages, use it; else, use as is
        if hasattr(prompt_value, 'to_messages'):
            prompt_for_model = prompt_value.to_messages()
        else:
            prompt_for_model = prompt_value
        # logger.info(f"Prompt for model: {prompt_for_model!r}")

        response = self.model.invoke(prompt_for_model)
        # logger.info(f"Model Response: {response!r}")

        label = response.content \
                    if hasattr(response, "content") \
                        else response

        logger.info(f"Label: {label}")

        return UserIntentEnum.OTHER \
            if label not in UserIntentEnum \
                else label

    def __call__(
        self, 
        **params
    ) -> str:
        return self.classify(**params)


class HuggingFaceClassificationModel(BaseClassificationModel):
    """Huggingface Classification Model"""

    def __init__(self) -> None:
        pass

    def __call__(self, **params) -> Any:
        pass


@deprecated("The class can not be used because there's no finetuned models")
class CohereClassificationModel(BaseClassificationModel):
    """ The class call classification API from Cohere"""

    def __init__(self, model: str) -> None:
        self.api = CohereAPI()
        self.model = model

    def __call__(self, **params) -> str:
        self.api.classify(model=self.model, **params)


class JinaClassificationModel(BaseClassificationModel):
    """ The class call classification API from Cohere"""

    def __init__(self, model: str) -> None:
        self.api = JinaAPI()
        self.model = model

    def __call__(self, **params) -> str:
        return self.api.classify(model=self.model, **params)


class ClassificationModel(BaseMLModel):
    """
        Classification Model 
    """

    def __init__(
        self, 
        model: str,
        framework: str,
        name: str,
        prompt_template: List[BaseMessage] | None = None,
        **kwargs
    ) -> None:
        """
            Parameters:
                model (str): the name of the model
                    If the model is Gemini, you need to provide `prompt_template` params with data type of List[BaseMessage]If the 
                    If the model is Huggingface model, it's not been provided yet, do not use it.
                framework (str): provide the source of the model (gemini, cohere, jina, ...)
                    Framework for Huggingface has not been provided
                name (str): just the name that you want to name your model
        """
        
        self.name = name
        
        if LLMModelEnum.GEMINI == framework:
            self.model = GeminiClassificationModel(
                model=model,
                prompt_template=prompt_template
            )
        elif LLMModelEnum.HUGGINGFACE  == framework:
            self.model = HuggingFaceClassificationModel(
                # model=model,
                # temperature=temperature,
                # max_tokens=max_tokens,
                # api_key=env_config.HUGGINGFACE_API_KEY
                # **kwargs
            )
        elif LLMModelEnum.COHERE  == framework:
            self.model = CohereClassificationModel(model=model)
        elif LLMModelEnum.JINA  == framework:
            self.model = JinaClassificationModel(model=model)
        else:
            raise ValueError("Model not supported")

        logger.info(f"Load chat model {model} succesfully")

    def __call__(self, *args: Any, **kwargs: Any) -> str:
        """
            The classification method

            Parameters:
                With the Jina and Cohere model, you need to provide inputs and labels
                If the model is Gemini, you need to provide parameters of the prompt template
        """

        return self.model(**kwargs)
from typing import Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_cohere.chat_models import ChatCohere

from src.enums import LLMModelEnum
from src.configs import env_config
from src.logger.logger import Logger
from src.utils.constants import DEFAULT_ERROR_MESSAGE
from src.ml_models.base import BaseChatModel, BaseMLModel


logger = Logger()


class ChatLLM(BaseMLModel):
    """
        Chat Model 
    """

    def __init__(
        self, 
        model: str,
        framework: str,
        name: str,
        temperature: float = 0.0,
        max_tokens: int = 1000,
        **kwargs
    ) -> None:

        """
            Parameters:
                model (str): the name of the model
                    If the model is Gemini, you need to provide `prompt_template` params with data type of List[BaseMessage]If the 
                    If the model is Cohere, you need to add prefix token "cohere/" to model name.
                    If the model is Jina, you just have to provide model name.
                    If the model is Huggingface model, it's not been provided yet, do not use it.
                framework (str): provide the source of the model (gemini, cohere, jina, ...)
                name (str): just the name that you want to name your model
        """

        self.model = None

        match framework:
            case LLMModelEnum.GEMINI:

                self.model = ChatGoogleGenerativeAI(
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    api_key=env_config.GOOGLE_API_KEY,
                    **kwargs
                )

            case LLMModelEnum.GROQ:
            
                self.model = ChatGroq(
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    api_key=env_config.GROQ_API_KEY,
                    **kwargs
                )

            case LLMModelEnum.HUGGINGFACE:

                llm = HuggingFacePipeline.from_model_id(
                    model_id=model,
                    task="text-generation",
                    token=env_config.HUGGINGFACE_API_KEY,
                    pipeline_kwargs=kwargs
                )

                self.model = ChatHuggingFace(
                    llm=llm
                )

            case LLMModelEnum.COHERE:

                self.model = ChatCohere(
                    temperature=temperature,
                    max_tokens=max_tokens,
                    cohere_api_key=env_config.COHERE_API_KEY,
                    **kwargs
                )
            
            case _:
                raise ValueError("Model not supported")

        logger.info(f"Load chat model {model} succesfully")

    def __call__(self, input: Any, **kwarg: Any) -> str:
        """
            The call method to call the LLM

            Parameters:
                input: can be a string or a value invoked by PromptTemplate

            Return the output of string
        """
        try:
            output = self.model.invoke(input).output

            logger.info(f"Output: {output.content}")

            return output
        except Exception as e:

            logger.error("Error happened: " + str(e))

            return DEFAULT_ERROR_MESSAGE


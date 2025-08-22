import json
from typing import Any, Dict
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage

from src.ml_models.chat_llm import ChatLLM
from src.logger.logger import Logger
from src.services.base import BaseService


logger = Logger()


class ExtractionService(BaseService):
    EXTRACTOR_SYSTEM_PROMPT = """You are an expert information extractor.  
    Your task is to analyze a user’s storytelling request and extract key structured features.  
    Do not create or imagine new content — only extract what is explicitly provided.  
    If a field is not mentioned, return an empty array or empty string for that field.  

    Always respond in valid JSON with the following schema:

    {
        "characters": [ "list of characters explicitly mentioned" ],
        "genres": [ "list of genres or styles requested" ],
        "conversations": [ "sample dialogues or quoted conversations provided by the user, include characters with there conversation" ],
        "context": "the setting, environment, or time period of the story",
        "description": "general description or summary of the story idea",
        "emotions": [ "list of emotions or moods expressed in the story" ]
    }
    """

    EXTRACTOR_USER_PROMPT = "{requirement}"

    llm_model = ChatLLM(
        model="gemini-2.5-flash",
        framework="gemini",
        name="Entity Extracting Model",
        
    )

    @classmethod
    def extract(cls, prompt: str) -> Dict[str, Any]:
        result = cls.llm_model([
            SystemMessage(content=cls.EXTRACTOR_SYSTEM_PROMPT),
            HumanMessage(content=cls.EXTRACTOR_USER_PROMPT.format(requirement=prompt))
        ])

        result = result.replace("```", "").replace("json", "").strip()

        logger.info(f"Result: {result}")

        json_response = json.loads(result)

        return json_response




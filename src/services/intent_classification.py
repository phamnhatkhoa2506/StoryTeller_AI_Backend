from langchain.prompts import ChatPromptTemplate

from src.ml_models import ClassificationModel
from src.services.base import BaseService


class IntentClassificationService(BaseService):

    INTENT_CLASSIFICATION_SYSTEM_PROMPT = """You are an expert intent classifier for storytelling tasks.  
    Your job is to analyze a user query and classify it into **exactly one** of the following intent labels:

    - "TELL": when the user wants a brand new fictional story to be told.  
    Examples: "Tell me a fairy tale about a dragon." / "Can you make a bedtime story for kids?"  

    - "CONTINUE": when the user wants to continue or extend a previously started story, based on earlier context.  
    Examples: "Continue the story about the pirate we discussed." / "What happens next in the tale?"  

    - "SUMMARY": when the user requests a summary of a story they provide.  
    Examples: "Summarize this story about the fox and the grapes." / "Give me a short recap of this tale."  

    - "OTHER": when the query is unrelated to storytelling (asking for facts, instructions, casual chat, etc.).  
    Examples: "What is the capital of France?" / "Explain how airplanes work."  

    ---

    ### Output format:
    Always respond with **only one of these labels**:  
    `TELL` | `CONTINUE` | `SUMMARY` | `OTHER`

    Do not output anything else.
    """

    INTENT_CLASSIFICATION_USER_PROMPT = "Classify this intent: {requirement}"

    model = ClassificationModel(
        model="gemini-2.5-flash",
        framework="gemini",
        name="Intent Classification Model",
        prompt_template=ChatPromptTemplate.from_messages([
            ("system", INTENT_CLASSIFICATION_SYSTEM_PROMPT),
            ("human", INTENT_CLASSIFICATION_USER_PROMPT)
        ])
    )

    @classmethod
    def classify(cls, prompt: str) -> str:
        return cls.model(requirement=prompt)
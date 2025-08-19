from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage

from src.ml_models import ChatLLM
from src.logger.logger import Logger 
from src.chains.other_chain.constants import OtherChainConstants


logger = Logger()


class OtherChain:
    """A chain help process ambigous intent of users"""

    def __init__(self,) -> None:
        self.model = ChatLLM(
            model="gemini-2.5-flash",
            framework="gemini",
            temperature=0.7,
            name="Other Intent Chat Model"
        )

        self.template = ChatPromptTemplate.from_messages([
            SystemMessage(content=OtherChainConstants.OTHER_CHAIN_SYSTEM_PROMPT),
            HumanMessage(content=OtherChainConstants.OTHER_CHAIN_USER_PROMPT)
        ])

    def answer(self, prompt: str) -> str:

        messages = self.template.invoke({"question": prompt})
        logger.info(f"Messages: {messages}")

        response = self.model.invoke(messages)

        logger.info(f"Response: {response}")
        
        if hasattr(response, "content"):
            return response.content

        return response
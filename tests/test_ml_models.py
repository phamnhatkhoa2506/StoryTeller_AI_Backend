import unittest
from unittest import TestCase
from langchain.prompts import ChatPromptTemplate

from src.ml_models import *
from src.configs import env_config


class TestMLModels(TestCase):

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
    
    @unittest.skip("Tested already")
    def test_gemini_chat_llms(self):
        model_name = "gemini-2.5-flash"
        gemini_chat_llm = ChatLLM(
            model=model_name,
            framework="gemini",
            name="gemini",
            temperature=0.5
        )

        prompt_template = "Haha"
        gemini_chat_llm.invoke(prompt_template)

    @unittest.skip("Not now")
    def test_groq_chat_llms(self):
        model_name = "llama-3.1-8b-instant"
        groq_chat_llm = ChatLLM(
            model=model_name,
            framework="groq",
            name="groq",
            temperature=0.5
        )

        prompt_template = "Haha"
        groq_chat_llm.invoke(prompt_template)
        
    @unittest.skip("Not now")
    def test_hf_chat_llms(self):
        model_name = "mistralai/Mistral-7B-Instruct-v0.2"
        # hf_chat_llm = ChatLLM(
        #     model=model_name,
        #     framework="hf",
        #     name="hf",
        #     temperature=0.5
        # )

    @unittest.skip("Not now")
    def test_cohere_chat_llms(self):
        model_name = "command-a-03-2025"
        cohere_chat_llm = ChatLLM(
            model=model_name,
            framework="cohere",
            name="gemini",
            temperature=0.5
        )
        
    @unittest.skip("Not now")
    def test_classifier_llms(self):
        pass

    @unittest.skip("Tested already")
    def test_gemini_embedding_model(self):
        model_name = env_config.GEMINI_EMBEDDING_MODEL
        embedding_model = EmbeddingModel(
            model=model_name,
            framework="gemini",
            name="Test Embedding Model"
        )

        texts = [
            "Hello",
            "Haha"
        ]

        result = embedding_model(
            documents=texts
        )

        print(result)

    @unittest.skip("Tested already")
    def test_cohere_embedding_model(self):
        model_name = env_config.COHERE_EMBEDDING_MODEL
        embedding_model = EmbeddingModel(
            model=model_name,
            framework="cohere",
            name="Test Embedding Model"
        )

        texts = [
            "Hello",
            "Haha"
        ]

        result = embedding_model(
            documents=texts,
            task="classification"
        )

        print(result)

    @unittest.skip("Tested already")
    def test_jina_embedding_model(self):
        model_name = env_config.JINA_EMBEDDING_MODEL
        embedding_model = EmbeddingModel(
            model=model_name,
            framework="jina",
            name="Test Embedding Model"
        )

        texts = [
            "Hello",
            "Haha"
        ]

        result = embedding_model(
            documents=texts,
            task="classification"
        )

        print(result)

    @unittest.skip("Tested already")
    def test_gemini_classification_model(self):
        model = ClassificationModel(
            model="gemini-2.5-flash",
            framework="gemini",
            name="Intent Classification Model",
            prompt_template=ChatPromptTemplate.from_messages([
                ("system", self.INTENT_CLASSIFICATION_SYSTEM_PROMPT),
                ("human", self.INTENT_CLASSIFICATION_USER_PROMPT)
            ])
        )

        queries = [
            """Hãy kể cho tôi một câu chuyện về bầy ong""",
            """Hello bạn khoẻ không""",
            """Hãy viết tiếp câu chuyện này cho tôi""",
            """Hãy tóm tắt câu chuyện này cho tôi"""
        ]

        for query in queries:
            print(f"Intent: {model(requirement=query)}")

    @unittest.skip("The method is deprecated")
    def test_cohere_classification_model(self):
        model = ClassificationModel(
            model=env_config.COHERE_EMBEDDING_MODEL,
            framework="cohere",
            name="Intent Classification Model",
        )

        inputs = [
            """Hãy kể cho tôi một câu chuyện về bầy ong""",
            """Hello bạn khoẻ không""",
            """Hãy viết tiếp câu chuyện này cho tôi""",
            """Hãy tóm tắt câu chuyện này cho tôi"""
        ]

        labels = [
            {
                "text": "Hãy kể",
                "label": "TELL"
            },
            {
                "text": "Hello",
                "label": "OTHER"
            },
            {
                "text": "Tiếp tục kể",
                "label": "CONTINUE"
            },
            {
                "text": "Tóm tắ",
                "label": "SUMMARY"
            }
        ]

        res = model(input=inputs, labels=labels)

    def test_jina_classification_model(self):
        model = ClassificationModel(
            model=env_config.JINA_EMBEDDING_MODEL,
            framework="jina",
            name="Intent Classification Model",
        )

        inputs = [
            """Hãy kể cho tôi một câu chuyện về bầy ong""",
            """Hello bạn khoẻ không""",
            """Hãy viết tiếp câu chuyện này cho tôi""",
            """Hãy tóm tắt câu chuyện này cho tôi"""
        ]

        labels = [
            "TELL",
            "CONTINUE",
            "OTHER",
            "SUMMARY"
        ]

        res = model(input=inputs, labels=labels)

        print(res)


if __name__ == "__main__":
    unittest.main()
import unittest
from unittest import TestCase

from src.ml_models.chat_llm import ChatLLM


class TestMLModels(TestCase):
    
    def test_chat_llms(self):
        model_name = "gemini-2.5-flash"
        gemini_chat_llm = ChatLLM(
            model=model_name,
            framework="gemini",
            name="gemini",
            temperature=0.5
        )

        model_name = "llama-3.1-8b-instant"
        groq_chat_llm = ChatLLM(
            model=model_name,
            framework="groq",
            name="groq",
            temperature=0.5
        )

        # model_name = "mistralai/Mistral-7B-Instruct-v0.2"
        # hf_chat_llm = ChatLLM(
        #     model=model_name,
        #     framework="hf",
        #     name="hf",
        #     temperature=0.5
        # )

        model_name = "command-a-03-2025"
        cohere_chat_llm = ChatLLM(
            model=model_name,
            framework="cohere",
            name="gemini",
            temperature=0.5
        )
        

        prompt_template = "Haha"
        gemini_chat_llm.invoke(prompt_template)
        groq_chat_llm.invoke(prompt_template)

    def test_classifier_llms(self):
        ...

    def test_embedding(self):
        ...


if __name__ == "__main__":
    unittest.main()
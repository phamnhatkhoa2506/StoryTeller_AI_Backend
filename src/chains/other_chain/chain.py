from src.ml_models import ChatLLM


class OtherChain:
    def __init__(self,) -> None:
        self.model = ChatLLM(
            model="gemini-2.5-pro",
            framework="gemini",
            temperature=0.7,
        )

    def answer(self, prompt: str) -> str:
        return self.model.invoke(prompt)
import unittest

from src.services import *


class TestServices(unittest.TestCase):

    preprocessing_service = PreprocessingService
    intent_classification_service = IntentClassificationService
    extraction_service = ExtractionService

    def test_preprocessing_service(self):
        string = """    ihjfgf    oljjjf.   
        Hello Ban nho djfhf
        dkdjhf dklj  f
        """

        string = self.preprocessing_service.remove_whitespace(string)
        print(string)
        string = self.preprocessing_service.normalize(string)
        print(string)
        lang = self.preprocessing_service.detect_language(string)
        print(lang)

    def test_intent_classification_service(self):
        queries = [
            """Hãy kể cho tôi một câu chuyện về bầy ong""",
            """Hello bạn khoẻ không""",
            """Hãy viết tiếp câu chuyện này cho tôi""",
            """Hãy tóm tắt câu chuyện này cho tôi"""
        ]

        for query in queries:
            print(f"Intent: {self.intent_classification_service.classify(query)}")

    def test_extraction_service(self):
        query = """
        I want a funny story about a pirate and a talking parrot in the 18th century. The pirate says 'Ahoy, matey!' and the parrot replies 'Treasure ahead!'. Make it adventurous and cheerful.
        """

        print(self.extraction_service.extract(query))


if __name__ == "__main__":
    unittest.main()
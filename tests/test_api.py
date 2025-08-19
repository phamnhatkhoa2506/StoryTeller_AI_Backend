import unittest as unt
from unittest import TestCase

from src.api import *


class TestApi(TestCase):

    def test_jina_api(self):
        jina_api = JinaAPI()

        classfication_model_name = 'jina-embeddings-v3'
        embedding_model_name = 'jina-embeddings-v3'
        reranking_model_name = 'jina-reranker-v2-base-multilingual'

        # Test classification
        inputs = [
            "Calculate the compound interest on a principal of $10,000 invested for 5 years at an annual rate of 5%, compounded quarterly.",
            "分析使用CRISPR基因编辑技术在人类胚胎中的伦理影响。考虑潜在的医疗益处和长期社会后果。",
            "AIが自意識を持つディストピアの未来を舞台にした短編小説を書いてください。人間とAIの関係や意識の本質をテーマに探求してください。",
            "Erklären Sie die Unterschiede zwischen Merge-Sort und Quicksort-Algorithmen in Bezug auf Zeitkomplexität, Platzkomplexität und Leistung in der Praxis.",
            "Write a poem about the beauty of nature and its healing power on the human soul.",
            "Translate the following sentence into French: The quick brown fox jumps over the lazy dog."
        ]
        labels = [
            "Simple task",
            "Complex reasoning",
            "Creative writing"
        ]
        jina_api.classify(
            model=classfication_model_name,
            input=inputs,
            labels=labels
        )

        # Test embedding
        input = [
            "Organic skincare for sensitive skin with aloe vera and chamomile: Imagine the soothing embrace of nature with our organic skincare range, crafted specifically for sensitive skin. Infused with the calming properties of aloe vera and chamomile, each product provides gentle nourishment and protection. Say goodbye to irritation and hello to a glowing, healthy complexion.",
        ]
        tasks = [
            "text-matching",
            "classification",
            "separation",
            "retrieval.query",
            "retrieval.passage"
        ]
        for task in tasks:
            res = jina_api.embed(
                model=embedding_model_name,
                input=input,
                task=task
            )

            self.assertIn("embedding", res[0])

        # Test reranking

    def test_cohere_api(self):
        cohere__classification_api = CohereAPI()


if __name__ == "__main__":
    unt.main()
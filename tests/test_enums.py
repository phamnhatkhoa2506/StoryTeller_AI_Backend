import unittest
from unittest import TestCase

from src.enums import *


class TestEnums(TestCase):
    """Test enums object"""

    def test_llm_model_enum(self):
        """Test LLM model enum"""

        self.assertEqual(LLMModelEnum.GEMINI, "gemini")
        self.assertEqual(LLMModelEnum.JINA, "jina")
        self.assertEqual(LLMModelEnum.GROQ, "groq")
        self.assertEqual(LLMModelEnum.HUGGINGFACE, "hf")

    def test_usage_type_enum(self):
        """Test usage type enum"""

        self.assertEqual(UsageTypeEnum.GUEST, "guest")
        self.assertEqual(UsageTypeEnum.NORMAL, "normal")
        self.assertEqual(UsageTypeEnum.PRO, "pro")
        

if __name__ == "__main__":
    unittest.main()
import unittest

from src.configs import *


class TestConfig(unittest.TestCase):
    
    def test_env_config(self):
        print(env_config.APP_URL)

    def test_ai_config(self):
        print(AIConfig.GEMINI_IMAGE_GENERATOR_CONFIG)
        print(AIConfig.GEMINI_SPEECH_GENERATOR_CONFIG)

    def test_vectordb_config(self):
        print(VectorDbConfig.CHROMA_SERVER_SETTINGS)


if __name__ == "__main__":
    unittest.main()
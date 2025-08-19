from io import BytesIO
from typing import Generator
from google import genai
from google.genai import types
from PIL import Image

from src.api.base import BaseAPI
from src.logger.logger import Logger


logger = Logger()


class GoogleGenAI_API(BaseAPI):
    """
        Google Generative API
    """

    def __init__(self, api_key: str) -> None:
        super(GoogleGenAI_API).__init__()

        self.client = genai.Client(api_key=api_key)

    def generate_image(
        self, 
        model: str,
        prompt: str,
        config: types.GenerateImagesConfig,
        **kwargs
    ) -> Generator[..., ..., Image] | None:
        """
            Generate image

            Parameters:
                model (str): the model you want
                prompt (str): prompt for AI
                config: configuration for image

            Return the generator of images or None if error happens
        """

        try:

            response = self.client.models.generate_images(
                model=model,
                prompt=prompt,
                config=config
            )

            for generated_image in response.generated_images:
                yield generated_image.image

        except Exception as e:
            logger.error("Error happed when call API for generating image")

            return None

    def generate_speech(
        self,
        model: str,
        prompt: str,
        config: types.GenerateContentConfig,
        **kwargs
    ) -> bytes | None:
        """
            Generate speech

            Parameters:
                model (str): the model you want
                prompt (str): prompt for AI
                config: configuration for audio, you can configure multicharacter for the audio

            Return the audio with byte data or None if error happens
        """

        try:

            response = self.client.models.generate_content(
                model=model,
                contents=prompt,
                config=config
            )

            data = response.candidates[0].content.parts[0].inline_data.data

            return data

        except Exception as e:
            
            logger.error("Error happed when call API for generating speech")

            return None
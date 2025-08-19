from google.genai import types


GEMINI_IMAGE_GENERATOR_CONFIG = types.GenerateImagesConfig(
    number_of_images=1,
)

GEMINI_SPEECH_GENRATOR_CONFIG = types.GenerateContentConfig(
    response_modalities=["AUDIO"],
    speech_config=types.SpeechConfig(
        voice_config=types.VoiceConfig(
            prebuilt_voice_config=types.PrebuiltVoiceConfig(
                voice_name='Kore',
            )
        )
    ),
)
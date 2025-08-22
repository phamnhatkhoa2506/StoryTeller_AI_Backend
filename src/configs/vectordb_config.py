from chromadb.config import Settings
from src.configs import env_config


class VectorDbConfig(object):
    CHROMA_SERVER_SETTINGS: Settings = Settings(
        chroma_client_auth_provider=env_config.CHROMA_SERVER_AUTHN_PROVIDER,
        chroma_client_auth_credentials="vnht1202:vnht1202"
    )
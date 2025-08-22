from typing import Any, Callable
from chromadb import CloudClient, HttpClient
from chromadb.config import Settings
from langchain_chroma import Chroma

from src.vectordb.base import BaseVectorDB
from src.logger.logger import Logger


logger = Logger()


class LocalChromaDB(BaseVectorDB):
    def __init__(
        self,
        collection_name: str,
        embedding_func: Callable[Any, Any],
        persist_directory: str,
        **kwargs
    ) -> None:
        """
            Parameters:
                collection_name (str): the name of collection
                embedding_func (func): embedding function of Gemini, Cohere, Jina or Huggingface
                persist_directory (str): the path to directory where store your data
        """

        self.__collection_name = collection_name
        
        self.client = self._get_connection(
            collection_name=collection_name,
            embedding_func=embedding_func,
            persist_directory=persist_directory
        )

    def _get_connection(self, **params) -> Chroma:
        """
            Connect to the chroma database local

            Parameters
                You need to provide these madantory parametes: host, port, collection_name, embedding_func
        
            Return:
                The Chroma Client connected to server
        """
        
        try:
            # Get params
            collection_name: str = params.get("collection_name")
            embedding_func: Callable[Any, Any] = params.get("embedding_func")
            persist_directory: str = params.get("persist_directory")

            # Get database
            db = Chroma(
                collection_name=collection_name,
                embedding_function=embedding_func,
                persist_directory=persist_directory
            )

            logger.info("Connecting to Chroma DB sucessfully.")

            return db

        except Exception as e:
            logger.info(f"Error when using local Chroma DB: {str(e)}")

            return None


class CloudChromaDB(BaseVectorDB):
    """The Chroma DB in Cloud"""

    def __init__(
        self,
        collection_name: str,
        embedding_func: Callable[Any, Any],
        tenant: str,
        api_key: str,
        database: str,
        **kwargs
    ) -> None:
        """
            Parameters:
                collection_name (str): the name of collection
                embedding_func (func): embedding function of Gemini, Cohere, Jina or Huggingface
                persist_directory (str): the path to directory where store your data
        """

        self.__collection_name = collection_name
        
        self.client = self._get_connection(
            collection_name=collection_name,
            embedding_func=embedding_func,
            tenant=tenant,
            database=database,
            api_key=api_key
        )

    def _get_connection(self, **params) -> Chroma:
        """
            Connect to the chroma database local

            Parameters
                You need to provide these madantory parametes: host, port, collection_name, embedding_func
        
            Return:
                The Chroma Client connected to Cloud
        """
        
        try:
            # Get params
            collection_name: str = params.get("collection_name")
            embedding_func: Callable[Any, Any] = params.get("embedding_func")
            tenant: str = params.get("tenant")
            database: str = params.get("database")
            api_key: str = params.get("api_key")

            # Get database
            chroma_client = CloudClient(
                tenant=tenant,
                database=database,
                api_key=api_key
            )

            db = Chroma(
                client=chroma_client,
                collection_name=collection_name,
                embedding_function=embedding_func,
            )

            logger.info("Connecting to Chroma DB sucessfully.")

            return db

        except Exception as e:
            logger.info(f"Error when connecting to Cloud Chroma DB: {str(e)}")

            return None


class ServerChromaDB(BaseVectorDB):
    """The chroma db run as a server"""

    def __init__(
        self,
        collection_name: str,
        embedding_func: Callable[Any, Any],
        host: str,
        port: int,
        settings: Settings,
        **kwargs
    ) -> None:

        """
            Parameters:
                collection_name (str): the name of collection
                embedding_func (func): embedding function of Gemini, Cohere, Jina or Huggingface
                host (str): the host of the server
                port (int): the port of the server
                settings (Settings): chroma settings
        """

        self.__collection_name = collection_name
        
        self.client = self._get_connection(
            host=host,
            port=port,
            collection_name=collection_name,
            embedding_func=embedding_func,
            settings=settings
        )

    def _get_connection(self, **params) -> Chroma:
        """
            Connect to the chroma database server

            Parameters
                You need to provide these madantory parametes: host, port, collection_name, embedding_func, settings for auth
        
            Return:
                The Chroma Client connected to server
        """

        try:
            # Get params
            host: int = params.get("host")
            port: str = params.get("port")
            collection_name: str = params.get("collection_name")
            embedding_func: Callable[Any, Any] = params.get("embedding_func")
            settings: Settings = params.get("settings")

            # Get database
            chroma_client = HttpClient(
                host=host,
                port=port,
                settings=settings
            )

            db = Chroma(
                client=chroma_client,
                collection_name=collection_name,
                embedding_function=embedding_func,
            )

            logger.info("Connecting to Chroma DB sucessfully.")

            return db

        except Exception as e:
            logger.info(f"Error when connecting to Server Chroma DB: {str(e)}")

            return None
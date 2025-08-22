import os
import unittest
import tempfile
import shutil
from uuid import uuid4
from dotenv import load_dotenv
from langchain.docstore.document import Document

from src.vectordb import *
from src.ml_models import EmbeddingModel
from src.configs import env_config, VectorDbConfig


class TestVectorDB(unittest.TestCase):

    load_dotenv()

    @unittest.skip("Not now")
    def test_local_chroma_db(self):
        tmpdir = tempfile.mkdtemp(prefix="vector_db_2")

        try:
            local_chroma = LocalChromaDB(
                collection_name="test_collection",
                embedding_func=...,
                persist_directory=tmpdir,
            )
          
        finally:
            del local_chroma
            shutil.rmtree(tmpdir, ignore_errors=True)

    @unittest.skip("Not now")
    def test_server_chroma_db(self):
        server_chroma = ServerChromaDB(
            collection_name="First",
            embedding_func=None,
            host=os.getenv("CHROMA_SERVER_HOST"),
            port=int(os.getenv("CHROMA_SERVER_PORT")),
            settings=VectorDbConfig.CHROMA_SERVER_SETTINGS
        )

    @unittest.skip("Not now")
    def test_cloud_chroma_db(self):
        cloud_chroma = CloudChromaDB(
            collection_name="First",
            embedding_func=None,
            tenant=os.getenv("CHROMA_CLOUD_TENANT"),
            api_key=os.getenv("CHROMA_API_KEY"),
            database=os.getenv("CHROMA_CLOUD_DATABASE")
        )

    @unittest.skip("Not now")
    def test_add_documents(self):
       
        tmpdir = tempfile.mkdtemp(prefix="vector_db")

        try:
            model_name = env_config.GEMINI_EMBEDDING_MODEL
            embedding_model = EmbeddingModel(
                model=model_name,
                framework="gemini",
                name="Test Embedding Model"
            )

            local_chroma = LocalChromaDB(
                collection_name="test_collection",
                embedding_func=embedding_model,
                persist_directory=tmpdir,
            )

            # Test add string first
            print("1. Test string and no indices")
            strings = [
                "Gekkfhgfjggg",
                "jghffgfgfgfg",
                "kdhfjgfgggg"
            ]
            res = local_chroma.add_documents(
                documents=strings
            )
            print(res)

            # Test add string first
            print("2. Test documents and no indices")
            strings = [
                Document(page_content="Gekkfhgfjggg"),
                Document(page_content="jghffgfgfgfg"),
                Document(page_content="kdhfjgfgggg")
            ]
            res = local_chroma.add_documents(
                documents=strings
            )
            print(res)

            # Test add string first
            print("3. Test string and indices")
            strings = [
                "Gekkfhgfjggg",
                "jghffgfgfgfg",
                "kdhfjgfgggg"
            ]
            ids = [uuid4().hex for _ in range(len(strings))]
            res = local_chroma.add_documents(
                documents=strings,
                ids=ids
            )
            print(res)

            # Test add string first
            print("4. Test documents and indices")
            strings = [
                Document(page_content="Gekkfhgfjggg"),
                Document(page_content="jghffgfgfgfg"),
                Document(page_content="kdhfjgfgggg")
            ]
            ids = [uuid4().hex for _ in range(len(strings))]
            res = local_chroma.add_documents(
                documents=strings,
                ids=ids
            )
            print(res)
          
        finally:
            del local_chroma
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_get_retriever(self):
        tmpdir = tempfile.mkdtemp(prefix="vector_db_2")

        try:
            local_chroma = LocalChromaDB(
                collection_name="test_collection",
                embedding_func=...,
                persist_directory=tmpdir,
            )

            retr = local_chroma.get_retriever()
          
        finally:
            del local_chroma
            shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
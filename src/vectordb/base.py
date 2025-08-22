from abc import ABC, abstractmethod
from ast import Dict
from typing import List, Optional
from langchain_chroma import Chroma
from langchain.docstore.document import Document
from langchain_core.vectorstores import VectorStoreRetriever

from src.exceptions import EmptyListError


class BaseVectorDB(ABC):
    """Base Class for Vector DB"""

    DEFAULT_TOP_K: int = 5
    
    @abstractmethod
    def _get_connection(self, **params) -> Chroma:
        pass

    def add_documents(
        self,
        documents: List[str | Document],
        ids: Optional[List[str]] = None,
        **kwargs
    ) -> List[str] | bool:
        """
            Add data to database

            Parameters:
                documents (List): the documents can be list of string or list of document instances
                ids (Optional): list of indices correspond to documents, may be not needed to provide
            
            Return:
                If ids is provided, return None. Otherwise, return the list of indices

            Raise:
                ValueError
        """

        if not documents:
            raise EmptyListError("You must provide text strings or documents, not empty list.")

        if ids and len(ids) != len(documents):
            raise ValueError("Number of indices must be equal to number of documents.")

        if isinstance(documents, list) and any(isinstance(d, str) for d in documents):
            documents = [
                Document(page_content=doc)
                    for doc in documents
            ]

        if ids:
            self.client.add_documents(
                documents=documents,
                ids=ids
            )
            return True

        else:
            ids = self.client.add_documents(
                documents=documents
            )

            return ids

    def similarity_search(
        self,
        query: str,
        k: int = DEFAULT_TOP_K,
        filter: Optional[Dict[str, str]] = None,
        **kwargs
    ) -> List[Document]:
        """
            Run similarity search with Chroma.

            Parameters:
                query: Query text to search for.
                k: Number of results to return. Defaults to 4.
                filter: Filter by metadata. Defaults to None.
                kwargs: Additional keyword arguments to pass to Chroma collection query.

            Returns:
                List of documents most similar to the query text.
        """

        return self.client.similarity_search(
            query=query,
            k=k,
            filter=filter,
            **kwargs
        )

    def get_retriever(
        self, 
        search_type: Optional[str],
        search_kwargs: Optional[Dict]
    ) -> VectorStoreRetriever:
        """
            Get retriever

            Parameters:
                search_type (Optional[str]): Defines the type of search that
                    the Retriever should perform. Can be "similarity" (default), "mmr", or "similarity_score_threshold".
                search_kwargs (Optional[Dict]): Keyword arguments to pass to the
                    search function. Can include things like:
                        k: Amount of documents to return (Default: 4)
                        score_threshold: Minimum relevance threshold
                            for similarity_score_threshold
                        fetch_k: Amount of documents to pass to MMR algorithm
                            (Default: 20)
                        lambda_mult: Diversity of results returned by MMR;
                            1 for minimum diversity and 0 for maximum. (Default: 0.5)
                        filter: Filter by document metadata

            Returns:
                VectorStoreRetriever: Retriever class for VectorStore.
        """

        return self.client.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs
        )

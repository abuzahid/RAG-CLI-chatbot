import os
from typing import List, Dict, Any, Optional
from langchain_chroma import Chroma
from langchain_core.documents import Document
from src.vectorstore.embedder import get_embedder
from src.config import get_config


class VectorStore:
    """Wrapper around ChromaDB for document storage and retrieval"""

    def __init__(
        self,
        persist_directory: Optional[str] = None,
        collection_name: Optional[str] = None,
    ):
        config = get_config()

        self.persist_directory = persist_directory or config.vectorstore.persist_directory
        self.collection_name = collection_name or config.vectorstore.collection_name

        # Ensure persist directory exists
        os.makedirs(self.persist_directory, exist_ok=True)

        self._db: Optional[Chroma] = None

    def _get_db(self) -> Chroma:
        """Lazy initialization of ChromaDB"""
        if self._db is None:
            embedder = get_embedder()
            self._db = Chroma(
                collection_name=self.collection_name,
                embedding_function=embedder,
                persist_directory=self.persist_directory,
            )
        return self._db

    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """Add documents to the vector store

        Args:
            documents: List of dicts with 'text' and optional 'metadata' keys
        """
        langchain_docs = []
        for doc in documents:
            metadata = doc.get("metadata", {})
            langchain_docs.append(
                Document(page_content=doc["text"], metadata=metadata)
            )

        db = self._get_db()
        db.add_documents(langchain_docs)

    def search(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Search for similar documents

        Args:
            query: Search query
            n_results: Number of results to return

        Returns:
            List of dicts with 'text', 'metadata', and 'score' keys
        """
        db = self._get_db()
        results = db.similarity_search_with_score(query, k=n_results)

        return [
            {
                "text": doc.page_content,
                "metadata": doc.metadata,
                "score": float(score),
            }
            for doc, score in results
        ]

    def clear(self) -> None:
        """Clear all documents from the collection"""
        db = self._get_db()
        # Delete all documents (not the collection itself)
        collection = db._collection
        collection.delete(where={})  # Delete all documents

        # Reset DB instance
        self._db = None

    def as_retriever(self, **kwargs):
        """Return a LangChain retriever"""
        db = self._get_db()
        return db.as_retriever(**kwargs)


def get_vectorstore() -> VectorStore:
    """Get the default vector store instance"""
    return VectorStore()

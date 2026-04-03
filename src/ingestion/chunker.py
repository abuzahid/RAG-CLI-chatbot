from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from src.config import get_config


class DocumentChunker:
    """Split documents into chunks for embedding and retrieval"""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    @classmethod
    def from_config(cls) -> "DocumentChunker":
        """Create chunker from configuration"""
        config = get_config()
        return cls(
            chunk_size=config.retrieval.chunk_size,
            chunk_overlap=config.retrieval.chunk_overlap,
        )

    def chunk_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Split documents into chunks

        Args:
            documents: List of dicts with 'text' and 'metadata' keys

        Returns:
            List of chunked documents with updated metadata
        """
        all_chunks = []

        for doc in documents:
            # Create LangChain document
            lc_doc = Document(page_content=doc["text"], metadata=doc.get("metadata", {}))

            # Split the document
            chunks = self._splitter.split_documents([lc_doc])

            # Add chunk index to metadata
            for i, chunk in enumerate(chunks):
                chunk.metadata["chunk_index"] = i
                chunk.metadata["chunk_total"] = len(chunks)

                all_chunks.append({
                    "text": chunk.page_content,
                    "metadata": chunk.metadata,
                })

        return all_chunks

import os
import shutil
import tempfile
import pytest
from unittest.mock import Mock, patch
from src.vectorstore.embedder import get_embedder, reset_embedder
from src.vectorstore.store import VectorStore
from src.config import reset_config


@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset singletons between tests"""
    reset_config()
    reset_embedder()
    yield
    reset_config()
    reset_embedder()


@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="No API key")
def test_embedder_creates_embeddings():
    """Test that embedder can create embeddings from text"""
    embedder = get_embedder()
    texts = ["Life insurance policy", "Term life coverage"]
    embeddings = embedder.embed_documents(texts)

    assert len(embeddings) == 2
    assert len(embeddings[0]) > 0  # Embedding vector should have dimensions
    assert all(isinstance(e, float) for e in embeddings[0])


@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="No API key")
def test_embedder_creates_query_embedding():
    """Test that embedder can create query embedding"""
    embedder = get_embedder()
    embedding = embedder.embed_query("What is term life insurance?")

    assert len(embedding) > 0
    assert all(isinstance(e, float) for e in embedding)


def test_embedder_returns_singleton():
    """Test that get_embedder returns same instance"""
    with patch('src.vectorstore.embedder.get_config') as mock_config:
        mock = Mock()
        mock.openai_api_key = "test-key"
        mock.embeddings.model = "text-embedding-3-small"
        mock_config.return_value = mock

        embedder1 = get_embedder()
        embedder2 = get_embedder()
        assert embedder1 is embedder2


@pytest.fixture
def temp_vectorstore(tmp_path):
    """Create a temporary vector store for testing"""
    with patch('src.config.get_config') as mock_config:
        mock = Mock()
        mock.openai_api_key = "test-key"
        mock.vectorstore.persist_directory = str(tmp_path / "chroma")
        mock.vectorstore.collection_name = "test_collection"
        mock_config.return_value = mock

        vs = VectorStore(
            persist_directory=str(tmp_path / "chroma"),
            collection_name="test_collection"
        )
        yield vs
        # Cleanup - ignore Windows file locking issues
        try:
            if os.path.exists(tmp_path / "chroma"):
                shutil.rmtree(tmp_path / "chroma", ignore_errors=True)
        except Exception:
            pass  # ChromaDB may keep files locked on Windows


def test_vectorstore_initialization(temp_vectorstore):
    """Test that vector store initializes correctly"""
    assert temp_vectorstore is not None
    assert temp_vectorstore.collection_name == "test_collection"


def test_vectorstore_as_retriever(temp_vectorstore):
    """Test that vector store can return a LangChain retriever"""
    retriever = temp_vectorstore.as_retriever(search_kwargs={"k": 2})
    assert retriever is not None

import os
import tempfile
import pytest
from unittest.mock import Mock, patch
from src.chain.prompts import get_qa_prompt, get_system_message
from src.chain.retrieval_chain import create_retrieval_chain, RetrievalChain
from src.chat.session import ChatSession
from src.vectorstore.store import VectorStore
from src.ingestion.chunker import DocumentChunker
from src.config import reset_config
from src.vectorstore.embedder import reset_embedder


@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset singletons between tests"""
    reset_config()
    reset_embedder()
    yield
    reset_config()
    reset_embedder()


def test_get_qa_prompt_returns_prompt_template():
    """Test that get_qa_prompt returns a PromptTemplate"""
    prompt = get_qa_prompt()

    assert prompt is not None
    # Should have placeholders for context and question
    assert "{context}" in str(prompt.template)
    assert "{question}" in str(prompt.template)


def test_get_system_message_from_config():
    """Test that system message comes from config"""
    system_msg = get_system_message()

    assert "life insurance support assistant" in system_msg.lower()


def test_qa_prompt_formats_correctly():
    """Test that QA prompt formats with context and question"""
    prompt = get_qa_prompt()

    formatted = prompt.format(
        context="Insurance covers death benefits.",
        question="What is life insurance?"
    )

    assert "Insurance covers death benefits" in formatted
    assert "What is life insurance?" in formatted


@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY") == "test-key", reason="No valid API key")
def test_retrieval_chain_initialization():
    """Test RetrievalChain class initialization"""
    session = ChatSession(max_history=5)

    with patch('src.config.get_config') as mock_config:
        mock = Mock()
        mock.openai_api_key = os.getenv("OPENAI_API_KEY")
        mock.llm.model = "gpt-4o-mini"
        mock.llm.temperature = 0.7
        mock.llm.max_tokens = 500
        mock.retrieval.top_k = 3
        mock.vectorstore.persist_directory = "./data/chroma"
        mock.vectorstore.collection_name = "insurance_knowledge"
        mock_config.return_value = mock

        chain = RetrievalChain(session=session)
        assert chain.session is session


@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY") == "test-key", reason="No valid API key")
def test_retrieval_chain_query():
    """Test that retrieval chain can process queries"""
    from src.ingestion.loader import DocumentLoader

    # Setup: add test documents
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch('src.config.get_config') as mock_config:
            mock = Mock()
            mock.openai_api_key = os.getenv("OPENAI_API_KEY")
            mock.llm.model = "gpt-4o-mini"
            mock.llm.temperature = 0.7
            mock.llm.max_tokens = 500
            mock.retrieval.top_k = 3
            mock.retrieval.chunk_size = 500
            mock.retrieval.chunk_overlap = 50
            mock.vectorstore.persist_directory = tmpdir
            mock.vectorstore.collection_name = "test"
            mock_config.return_value = mock

            vs = VectorStore(persist_directory=tmpdir, collection_name="test")

            # Add test documents
            docs = [{"text": "Term life insurance provides coverage for a specific period.", "metadata": {}}]
            chunker = DocumentChunker(chunk_size=500, chunk_overlap=50)
            chunks = chunker.chunk_documents(docs)
            vs.add_documents(chunks)

            session = ChatSession(max_history=5)
            chain = RetrievalChain(session=session, vectorstore=vs)

            response = chain.query("What is term life insurance?")

            assert response is not None
            assert len(response) > 0


@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY") == "test-key", reason="No valid API key")
def test_retrieval_chain_includes_history():
    """Test that chain includes conversation history"""
    session = ChatSession(max_history=5)
    session.add_user_message("What is whole life insurance?")
    session.add_ai_message("Whole life insurance provides lifetime coverage.")

    with patch('src.config.get_config') as mock_config:
        mock = Mock()
        mock.openai_api_key = os.getenv("OPENAI_API_KEY")
        mock.llm.model = "gpt-4o-mini"
        mock.llm.temperature = 0.7
        mock.llm.max_tokens = 500
        mock.retrieval.top_k = 3
        mock.conversation.system_message = "You are a helpful assistant."
        mock.vectorstore.persist_directory = "./data/chroma"
        mock.vectorstore.collection_name = "insurance_knowledge"
        mock_config.return_value = mock

        chain = RetrievalChain(session=session)

        # Chain should have access to history
        assert len(chain.session.history) == 2


def test_retrieval_chain_handles_invalid_api_key(monkeypatch):
    """Test that chain handles invalid API key gracefully"""
    from src.vectorstore.store import VectorStore

    monkeypatch.setenv("OPENAI_API_KEY", "invalid-key")

    tmpdir = tempfile.mkdtemp()
    try:
        with patch('src.config.get_config') as mock_config:
            mock = Mock()
            mock.openai_api_key = "invalid-key"
            mock.llm.model = "gpt-4o-mini"
            mock.llm.temperature = 0.7
            mock.llm.max_tokens = 500
            mock.retrieval.top_k = 3
            mock.conversation.system_message = "You are a helpful assistant."
            mock.vectorstore.persist_directory = tmpdir
            mock.vectorstore.collection_name = "test"
            mock_config.return_value = mock

            vs = VectorStore(persist_directory=tmpdir, collection_name="test")
            session = ChatSession(max_history=5)

            # Chain initialization succeeds
            chain = RetrievalChain(session=session, vectorstore=vs)
            assert chain is not None

            # Query should raise error due to invalid API key
            with pytest.raises(Exception):
                chain.query("test question")
    finally:
        # Cleanup
        try:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)
        except:
            pass

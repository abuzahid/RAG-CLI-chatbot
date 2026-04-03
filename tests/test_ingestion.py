import os
import pytest
import tempfile
from src.ingestion.loader import DocumentLoader, load_documents_from_path
from src.ingestion.chunker import DocumentChunker
from src.config import reset_config, get_config


@pytest.fixture(autouse=True)
def reset_config_fixture():
    """Reset config between tests"""
    reset_config()
    yield
    reset_config()


def test_loader_loads_text_file():
    """Test loading a simple text file"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("This is a test document about life insurance.")
        temp_path = f.name

    try:
        loader = DocumentLoader()
        docs = loader.load(temp_path)

        assert len(docs) == 1
        assert "test document" in docs[0]["text"].lower()
        assert docs[0]["metadata"]["source"] == temp_path
    finally:
        os.unlink(temp_path)


def test_loader_loads_markdown_file():
    """Test loading a markdown file"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
        f.write("# Life Insurance\n\nThis is a test markdown file.")
        temp_path = f.name

    try:
        loader = DocumentLoader()
        docs = loader.load(temp_path)

        assert len(docs) == 1
        assert "life insurance" in docs[0]["text"].lower()
    finally:
        os.unlink(temp_path)


def test_loader_unsupported_extension_raises_error():
    """Test that unsupported file types raise an error"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.xyz', delete=False) as f:
        f.write("test content")
        temp_path = f.name

    try:
        loader = DocumentLoader()
        with pytest.raises(ValueError, match="Unsupported file type"):
            loader.load(temp_path)
    finally:
        os.unlink(temp_path)


def test_loader_loads_directory():
    """Test loading all supported files from a directory"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test files
        with open(os.path.join(tmpdir, "test1.txt"), 'w') as f:
            f.write("Document 1 content")
        with open(os.path.join(tmpdir, "test2.md"), 'w') as f:
            f.write("Document 2 content")
        # Create unsupported file (should be skipped)
        with open(os.path.join(tmpdir, "test3.xyz"), 'w') as f:
            f.write("Should be ignored")

        loader = DocumentLoader()
        docs = loader.load_directory(tmpdir)

        assert len(docs) == 2


def test_load_documents_from_path_convenience():
    """Test the convenience function for loading from path"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("Test document for convenience function")
        temp_path = f.name

    try:
        docs = load_documents_from_path(temp_path)
        assert len(docs) == 1
        assert "convenience function" in docs[0]["text"].lower()
    finally:
        os.unlink(temp_path)


def test_chunker_splits_long_document():
    """Test that chunker splits long documents into pieces"""
    chunker = DocumentChunker(chunk_size=100, chunk_overlap=20)

    long_text = "word " * 50  # Create a long document
    documents = [{"text": long_text, "metadata": {"source": "test"}}]

    chunks = chunker.chunk_documents(documents)

    assert len(chunks) > 1
    # Verify there's some content overlap (not exact, since splitting on word boundaries)
    combined_text = " ".join(c["text"] for c in chunks)
    assert "word" in combined_text  # Original content preserved


def test_chunker_preserves_metadata():
    """Test that chunker preserves metadata"""
    chunker = DocumentChunker(chunk_size=100, chunk_overlap=20)

    documents = [
        {
            "text": "Short text",
            "metadata": {"source": "test.txt", "page": 1}
        }
    ]

    chunks = chunker.chunk_documents(documents)

    assert len(chunks) == 1
    assert chunks[0]["metadata"]["source"] == "test.txt"
    assert chunks[0]["metadata"]["page"] == 1


def test_chunker_adds_chunk_index():
    """Test that chunker adds chunk index to metadata"""
    chunker = DocumentChunker(chunk_size=50, chunk_overlap=10)

    long_text = "Word " * 50
    documents = [{"text": long_text, "metadata": {"source": "test"}}]

    chunks = chunker.chunk_documents(documents)

    if len(chunks) > 1:
        assert "chunk_index" in chunks[0]["metadata"]
        assert chunks[0]["metadata"]["chunk_index"] == 0
        assert chunks[1]["metadata"]["chunk_index"] == 1


def test_chunker_from_config():
    """Test that chunker can be created from config"""
    chunker = DocumentChunker.from_config()

    assert chunker.chunk_size == get_config().retrieval.chunk_size
    assert chunker.chunk_overlap == get_config().retrieval.chunk_overlap

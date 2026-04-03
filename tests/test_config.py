import os
import pytest
from src.config import Config, get_config, reset_config


@pytest.fixture(autouse=True)
def reset_config_fixture():
    """Reset config between tests"""
    reset_config()
    yield
    reset_config()


def test_config_loads_from_yaml():
    """Test that config loads from YAML file"""
    config = Config()
    assert config.llm.model == "gpt-4o-mini"
    assert config.llm.temperature == 0.7
    assert config.vectorstore.collection_name == "insurance_knowledge"


def test_config_loads_openai_key_from_env(monkeypatch):
    """Test that OpenAI API key is loaded from environment"""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-123")
    reset_config()
    config = Config()
    assert config.openai_api_key == "test-key-123"


def test_config_missing_api_key_raises_error(monkeypatch):
    """Test that missing API key raises appropriate error"""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    reset_config()
    with pytest.raises(ValueError, match="OPENAI_API_KEY not found"):
        Config()


def test_get_config_returns_singleton(monkeypatch):
    """Test that get_config returns the same instance"""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    reset_config()
    config1 = get_config()
    config2 = get_config()
    assert config1 is config2


def test_config_retrieval_settings():
    """Test retrieval settings are accessible"""
    config = Config()
    assert config.retrieval.chunk_size == 1000
    assert config.retrieval.chunk_overlap == 200
    assert config.retrieval.top_k == 3

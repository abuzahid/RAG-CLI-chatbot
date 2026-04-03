from langchain_openai import OpenAIEmbeddings
from src.config import get_config

_embedder_instance = None


def get_embedder():
    """Get singleton embedder instance"""
    global _embedder_instance
    if _embedder_instance is None:
        config = get_config()
        _embedder_instance = OpenAIEmbeddings(
            model=config.embeddings.model,
            api_key=config.openai_api_key,
        )
    return _embedder_instance


def reset_embedder():
    """Reset embedder instance (useful for testing)"""
    global _embedder_instance
    _embedder_instance = None

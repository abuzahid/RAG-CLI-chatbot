import os
from dataclasses import dataclass, field
from typing import Optional
import yaml
from dotenv import load_dotenv

load_dotenv()


@dataclass
class LLMConfig:
    provider: str = "openai"
    model: str = "gpt-4o-mini"
    temperature: float = 0.7
    max_tokens: int = 500


@dataclass
class EmbeddingsConfig:
    provider: str = "openai"
    model: str = "text-embedding-3-small"


@dataclass
class VectorStoreConfig:
    type: str = "chromadb"
    persist_directory: str = "./data/chroma"
    collection_name: str = "insurance_knowledge"


@dataclass
class RetrievalConfig:
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k: int = 3


@dataclass
class ConversationConfig:
    max_history: int = 10
    system_message: str = ""  # Will be loaded from YAML


@dataclass
class Config:
    llm: LLMConfig = field(default_factory=LLMConfig)
    embeddings: EmbeddingsConfig = field(default_factory=EmbeddingsConfig)
    vectorstore: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    conversation: ConversationConfig = field(default_factory=ConversationConfig)
    openai_api_key: Optional[str] = field(default=None)

    def __post_init__(self):
        # Load YAML config if it exists
        config_path = os.path.join(os.getcwd(), "config.yaml")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                yaml_config = yaml.safe_load(f) or {}

            # Update configs from YAML
            if "llm" in yaml_config:
                self.llm = LLMConfig(**yaml_config["llm"])
            if "embeddings" in yaml_config:
                self.embeddings = EmbeddingsConfig(**yaml_config["embeddings"])
            if "vectorstore" in yaml_config:
                self.vectorstore = VectorStoreConfig(**yaml_config["vectorstore"])
            if "retrieval" in yaml_config:
                self.retrieval = RetrievalConfig(**yaml_config["retrieval"])
            if "conversation" in yaml_config:
                self.conversation = ConversationConfig(**yaml_config["conversation"])

        # Load API key from environment
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")


_config_instance: Optional[Config] = None


def get_config() -> Config:
    """Get singleton config instance"""
    global _config_instance
    if _config_instance is None:
        _config_instance = Config()
    return _config_instance


def reset_config():
    """Reset config instance (useful for testing)"""
    global _config_instance
    _config_instance = None

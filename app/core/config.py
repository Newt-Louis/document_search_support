import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass(frozen=True)
class AppConfig:
    # Qdrant
    QDRANT_URL: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "company_docs")

    # Paths
    UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", "./data/uploads")
    CACHE_DIR: str = os.getenv("CACHE_DIR", "./data/cache")

    # Embedding
    EMBED_MODEL: str = os.getenv("EMBED_MODEL", "intfloat/multilingual-e5-large")
    EMBED_CACHE_DIR: str = os.getenv("EMBED_CACHE_DIR", "./data/cache/embeddings/multilingual-e5-large")

    # LLM (Ollama)
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3.2")
    OLLAMA_TIMEOUT: float = float(os.getenv("OLLAMA_TIMEOUT", "360"))
    OLLAMA_TEMPERATURE: float = float(os.getenv("OLLAMA_TEMPERATURE", "0.1"))
    OLLAMA_NUM_CTX: int = int(os.getenv("OLLAMA_NUM_CTX", "4096"))

def get_config() -> AppConfig:
    return AppConfig()
import os
from dataclasses import dataclass
from dotenv import load_dotenv
from pathlib import Path
from functools import lru_cache

load_dotenv()

@dataclass(frozen=True)
class AppConfig:
    # Base Dir
    BASE_DIR = Path(__file__).parent.parent.parent

    # Qdrant
    QDRANT_URL: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "company_docs")

    # Paths
    UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", str(BASE_DIR/"data"/"uploads"))
    CACHE_DIR: str = os.getenv("CACHE_DIR", str(BASE_DIR/"data"/"cache"))
    LOG_DIR: str = os.getenv("LOG_DIR", str(BASE_DIR/"data"/"logs"))

    # Embedding
    EMBED_MODEL: str = os.getenv("EMBED_MODEL", "intfloat/multilingual-e5-small")
    EMBED_CACHE_DIR: str = os.getenv("EMBED_CACHE_DIR", str(BASE_DIR/"data"/"cache"/"multilingual-e5-small"))

    # LLM (Ollama) - Config
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3.2")
    OLLAMA_TIMEOUT: float = float(os.getenv("OLLAMA_TIMEOUT", "360"))
    OLLAMA_NUM_CTX: int = int(os.getenv("OLLAMA_NUM_CTX", "4096"))

    # LLM - Personality
    OLLAMA_TEMPERATURE: float = float(os.getenv("OLLAMA_TEMPERATURE", "0.1"))
    OLLAMA_TOP_P: float = float(os.getenv("OLLAMA_TOP_P", "0.15"))
    OLLAMA_NUM_PREDICT: int = int(os.getenv("OLLAMA_NUM_PREDICT", "10"))
    OLLAMA_REPEAT_PENALTY: float = float(os.getenv("OLLAMA_REPEAT_PENALTY", "0.1"))

    def __post_init__(self):
        """Tạo các thư mục nếu chưa tồn tại"""
        for dir_path in [self.UPLOAD_DIR, self.CACHE_DIR, self.LOG_DIR]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)

@lru_cache
def get_config() -> AppConfig:
    cfg = AppConfig()
    cfg.__post_init__()
    return cfg
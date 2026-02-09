from contextlib import asynccontextmanager
import qdrant_client
from qdrant_client import AsyncQdrantClient
from llama_index.core import VectorStoreIndex, Settings
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.llms.ollama import Ollama

from app.core.config import get_config
from app.services.rag.engine import get_query_engine, KnowledgeBaseEmptyError

@asynccontextmanager
async def lifespan(app):
    cfg = get_config()
    print(">>> 🚀 Booting AI Server...")

    # Embed model (FastEmbed ONNX CPU)
    Settings.embed_model = FastEmbedEmbedding(
        model_name=cfg.EMBED_MODEL,
        cache_dir=cfg.EMBED_CACHE_DIR,
    )

    # LLM (Ollama)
    Settings.llm = Ollama(
        model=cfg.OLLAMA_MODEL,
        base_url=cfg.OLLAMA_BASE_URL,
        request_timeout=cfg.OLLAMA_TIMEOUT,
        temperature=cfg.OLLAMA_TEMPERATURE,
        additional_kwargs={"num_ctx": cfg.OLLAMA_NUM_CTX},
    )

    # Qdrant client + vector store
    client = qdrant_client.QdrantClient(url=cfg.QDRANT_URL)
    aclient = qdrant_client.AsyncQdrantClient(url=cfg.QDRANT_URL)
    vector_store = QdrantVectorStore(client=client, aclient=aclient, collection_name=cfg.COLLECTION_NAME)

    # Try build query_engine (nếu DB có dữ liệu)
    app.state.cfg = cfg
    app.state.qdrant_client = client
    app.state.qdrant_aclient = aclient
    app.state.vector_store = vector_store

    # Cache / state cho RAG
    app.state.index = None
    app.state.query_engine_json = None
    app.state.query_engine_stream = None

    try:
        _ = get_query_engine(app, streaming=False, top_k=3)
        _ = get_query_engine(app, streaming=True, top_k=3)
        print(">>> ✅ AI Engine Ready!")
    except Exception as e:
        print(f">>> ⚠️ DB empty / cannot init index yet: {e}")

    yield

    print(">>> 🛑 Server shutting down...")
    try:
        await aclient.close()  # Đóng kết nối async
        client.close()
    except:
        pass
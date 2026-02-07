from contextlib import asynccontextmanager
import qdrant_client
from llama_index.core import VectorStoreIndex, Settings
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.llms.ollama import Ollama

from app.core.config import get_config

@asynccontextmanager
async def lifespan(app):
    cfg = get_config()
    print(">>> 🚀 Booting AI Server...")

    # 1) Embed model (FastEmbed ONNX CPU)
    Settings.embed_model = FastEmbedEmbedding(
        model_name=cfg.EMBED_MODEL,
        cache_dir=cfg.EMBED_CACHE_DIR,
    )

    # 2) LLM (Ollama)
    Settings.llm = Ollama(
        model=cfg.OLLAMA_MODEL,
        base_url=cfg.OLLAMA_BASE_URL,
        request_timeout=cfg.OLLAMA_TIMEOUT,
        temperature=cfg.OLLAMA_TEMPERATURE,
        additional_kwargs={"num_ctx": cfg.OLLAMA_NUM_CTX},
    )

    # 3) Qdrant client + vector store
    client = qdrant_client.QdrantClient(url=cfg.QDRANT_URL)
    vector_store = QdrantVectorStore(client=client, collection_name=cfg.COLLECTION_NAME)

    # 4) Try build query_engine (nếu DB có dữ liệu)
    app.state.cfg = cfg
    app.state.qdrant_client = client
    app.state.vector_store = vector_store
    app.state.query_engine = None

    try:
        index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
        app.state.query_engine = index.as_query_engine(streaming=True, similarity_top_k=2)
        print(">>> ✅ AI Engine Ready!")
    except Exception as e:
        print(f">>> ⚠️ DB empty / cannot init index yet: {e}")

    yield

    print(">>> 🛑 Server shutting down...")
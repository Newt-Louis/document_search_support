# app/services/rag/engine.py
import json
import logging
from typing import Any, Dict, List, Optional

from llama_index.core import VectorStoreIndex
from app.services.rag.prompts import QA_TEMPLATE

logger = logging.getLogger(__name__)

class KnowledgeBaseEmptyError(RuntimeError):
    """Ném ra khi Qdrant chưa có dữ liệu / chưa load được index."""
    pass

def sse_event(event: str, data: Dict[str, Any]) -> str:
    """
    SSE format: event: <name>\n data: <json>\n\n
    (Server-Sent Events)
    """
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

def _extract_sources(resp, max_sources: int = 5) -> List[Dict[str, Any]]:
    """
    LlamaIndex response có thể có resp.source_nodes.
    Mình lấy ra text+metadata để trả JSON cho client.
    """
    sources: List[Dict[str, Any]] = []
    source_nodes = getattr(resp, "source_nodes", [])
    if not source_nodes: return []
    for sn in source_nodes[:max_sources]:
        node = getattr(sn, "node", None)
        if node is None: continue
        sources.append(
            {
                "score": getattr(sn, "score", None),
                "text": (getattr(node, "text", "") or "")[:1200],
                "metadata": getattr(node, "metadata", {}) or {},
            }
        )
    return sources


def _ensure_index(app) -> VectorStoreIndex:
    """
    Lấy con trỏ chỉ mục được đánh dấu từ qdrant
    Ưu tiên dùng app.state.index nếu có
    """
    index = getattr(app.state, "index", None)
    if index is not None:
        return index

    vector_store = getattr(app.state, "vector_store", None)
    if vector_store is None:
        raise RuntimeError("vector_store not initialized (check lifespan)")

    try:
        index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
        app.state.index = index
        return index
    except Exception as e:
        raise KnowledgeBaseEmptyError(str(e))


def get_query_engine(app, *, streaming: bool, top_k: int = 3):
    """
    Cache query_engine theo mode để tối ưu truy vấn:
    - chat_query_engine_json
    - chat_query_engine_stream
    """
    cache_key = "query_engine_stream" if streaming else "query_engine_json"
    qe = getattr(app.state, cache_key, None)
    if qe is not None:
        return qe

    index = _ensure_index(app)

    qe = index.as_query_engine(
        streaming=streaming,
        similarity_top_k=top_k,
        text_qa_template=QA_TEMPLATE,
        vector_store_kwargs={"aclient": app.state.qdrant_aclient}
    )
    setattr(app.state, cache_key, qe)
    return qe


def invalidate_engines(app):
    """Gọi sau ingest để query dùng index mới."""
    app.state.query_engine_json = None
    app.state.query_engine_stream = None


def set_index(app, index: VectorStoreIndex):
    """Gọi sau ingest để lưu index mới + clear cache engines."""
    app.state.index = index
    invalidate_engines(app)

async def query_json(app, question: str, *, top_k: int = 3) -> Dict[str, Any]:
    qe = get_query_engine(app, streaming=False, top_k=top_k)
    resp = await qe.aquery(question)

    return {
        "answer": str(resp),
        "sources": _extract_sources(resp),
        "meta": {"top_k": top_k, "streaming": False},
    }


async def query_sse_generator(app, question: str, *, top_k: int = 3):
    """
    Generator SSE: start -> token* -> done|error
    """
    qe = get_query_engine(app, streaming=True, top_k=top_k)

    yield sse_event("start", {"ok": True})
    try:
        resp = await qe.aquery(question)
        full = []
        async for token in resp.async_response_gen():
            full.append(token)
            yield sse_event("token", {"delta": token})

        yield sse_event(
            "done",
            {
                "answer": "".join(full),
                "sources": _extract_sources(resp),
                "meta": {"top_k": top_k, "streaming": True},
            },
        )
    except Exception as e:
        logger.exception(f"Chat Error: {e}")
        yield sse_event("error", {"message": str(e)})

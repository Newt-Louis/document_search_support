import qdrant_client, json
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.templating import Jinja2Templates
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.prompts import PromptTemplate
from pydantic import BaseModel

router = APIRouter(tags=["chat"])
templates = Jinja2Templates(directory="app/static/chat_widget")

class ChatRequest(BaseModel):
    question: str

class SourceItem(BaseModel):
    score: Optional[float] = None
    text: str
    metadata: Dict[str, Any] = {}

class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceItem] = []
    meta: Dict[str, Any] = {}

def sse_event(event: str, data: Dict[str, Any]) -> str:
    """
    SSE format: event: <name>\n data: <json>\n\n
    (Server-Sent Events)
    """
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"



# ---------- Prompt template (giống query.py) ----------

QA_PROMPT_STR = (
    "Bạn là trợ lý AI nội bộ. Dưới đây là thông tin ngữ cảnh (context) lấy từ tài liệu công ty:\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Dựa trên ngữ cảnh trên (và CHỈ dựa trên ngữ cảnh đó), hãy trả lời câu hỏi: {query_str}\n\n"
    "Yêu cầu bắt buộc:\n"
    "- Bắt đầu câu trả lời bằng cụm từ: 'Theo như thông tin tôi tìm được từ tài liệu nội bộ...'\n"
    "- Nếu không tìm thấy thông tin trong ngữ cảnh, hãy nói: 'Xin lỗi, tài liệu hiện tại không chứa thông tin này.'\n"
    "- Kết thúc bằng: 'Đây là tất cả thông tin tôi có được, mong sẽ giúp ích cho bạn.'"
)
QA_TEMPLATE = PromptTemplate(QA_PROMPT_STR)


# ---------- Helpers ----------

def _extract_sources(resp, max_sources: int = 5) -> List[Dict[str, Any]]:
    """
    LlamaIndex response có thể có resp.source_nodes.
    Mình lấy ra text+metadata để trả JSON cho client.
    """
    sources: List[Dict[str, Any]] = []
    for sn in getattr(resp, "source_nodes", [])[:max_sources]:
        node = getattr(sn, "node", None)
        if node is None:
            continue
        sources.append(
            {
                "score": getattr(sn, "score", None),
                "text": (getattr(node, "text", "") or "")[:1200],
                "metadata": getattr(node, "metadata", {}) or {},
            }
        )
    return sources


def get_query_engine(request: Request, *, streaming: bool, top_k: int = 3):
    """
    Dùng chung cho cả JSON và SSE.
    - Nếu app.state.query_engine chưa có (DB rỗng lúc boot), thử load lại từ Qdrant.
    - Lưu lại vào app.state để các request sau dùng luôn.
    - Nếu streaming=True thì tạo engine streaming.
    """
    cfg = request.app.state.cfg

    # cache theo mode để không bị đè lẫn nhau giữa json và stream
    cache_key = "query_engine_stream" if streaming else "query_engine_json"
    qe = getattr(request.app.state, cache_key, None)
    if qe is not None:
        return qe

    # Tạo lại index từ vector_store (nhẹ)
    try:
        client = qdrant_client.QdrantClient(url=cfg.QDRANT_URL)
        vector_store = QdrantVectorStore(client=client, collection_name=cfg.COLLECTION_NAME)
        index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

        qe = index.as_query_engine(
            streaming=streaming,
            similarity_top_k=top_k,
            text_qa_template=QA_TEMPLATE,
        )

        setattr(request.app.state, cache_key, qe)
        return qe
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Hệ thống chưa có dữ liệu hoặc không thể load index: {e}")

@router.get("/chat")
async def get_chat_ui(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@router.post("/chat",response_model=ChatResponse)
async def chat_endpoint(request: Request, payload: ChatRequest):
    """
    Trả JSON sau khi chạy xong.
    """
    qe = get_query_engine(request, streaming=False, top_k=3)

    resp = qe.query(payload.question)

    answer = str(resp)  # LlamaIndex response -> string
    sources = _extract_sources(resp)

    return {
        "answer": answer,
        "sources": sources,
        "meta": {
            "top_k": 3,
            "streaming": False,
        },
    }

@router.post("/chat/stream")
async def chat_sse(request: Request, payload: ChatRequest):
    """
    SSE streaming token. Client đọc event-stream và append token.
    Trả event:
        - start
        - token (delta)
        - done (answer + sources)
        - error
    """
    cfg = request.app.state.cfg
    query_engine = request.app.state.query_engine

    if query_engine is None:
        # reload attempt nếu boot chưa có data
        try:
            client = qdrant_client.QdrantClient(url=cfg.QDRANT_URL)
            vector_store = QdrantVectorStore(client=client, collection_name=cfg.COLLECTION_NAME)
            index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
            query_engine = index.as_query_engine(streaming=True, similarity_top_k=2)
            request.app.state.query_engine = query_engine
        except Exception:
            raise HTTPException(status_code=500, detail="Hệ thống chưa có dữ liệu. Vui lòng upload tài liệu trước.")

    def generator():
        yield sse_event("start", {"ok": True})

        try:
            resp = qe.query(payload.question)

            full_parts: List[str] = []
            for token in resp.response_gen:
                full_parts.append(token)
                yield sse_event("token", {"delta": token})

            sources = _extract_sources(resp)
            yield sse_event(
                "done",
                {
                    "answer": "".join(full_parts),
                    "sources": sources,
                    "meta": {"top_k": 3, "streaming": True},
                },
            )
        except Exception as e:
            yield sse_event("error", {"message": str(e)})

    return StreamingResponse(generator(), media_type="text/event-stream")

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from app.services.rag.engine import (
    KnowledgeBaseEmptyError,
    query_json,
    query_sse_generator,
)

router = APIRouter(tags=["chat"])
templates = Jinja2Templates(directory="app/static/chat_widget")

class ChatRequest(BaseModel):
    question: str

@router.get("/chat")
async def get_chat_ui(request: Request):
    """
    # UI test / demo cho frontend
    """
    return templates.TemplateResponse("index.html", {"request": request})

@router.post("/chat")
async def chat_json_endpoint(request: Request, payload: ChatRequest):
    """
    Trả JSON sau khi chạy xong.
    """
    try:
        return query_json(request.app, payload.question, top_k=3)
    except KnowledgeBaseEmptyError:
        raise HTTPException(status_code=500, detail="Knowledge base đang rỗng. Hãy ingest tài liệu trước.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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
    try:
        gen = query_sse_generator(request.app, payload.question, top_k=3)
        headers = {
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # tránh buffering nếu có reverse proxy
        }
        return StreamingResponse(gen, media_type="text/event-stream", headers=headers)
    except KnowledgeBaseEmptyError:
        raise HTTPException(status_code=500, detail="Knowledge base đang rỗng. Hãy ingest tài liệu trước.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

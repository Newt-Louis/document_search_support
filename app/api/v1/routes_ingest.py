import logging
from fastapi import APIRouter, UploadFile, File, HTTPException, Request, Depends
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from app.services.rag.ingest import IngestService

router = APIRouter(tags=["ingest"])
logger = logging.getLogger(__name__)
templates = Jinja2Templates(directory="app/static/upload_widget")

def get_ingest_service(request: Request) -> IngestService:
    cfg = request.app.state.cfg
    vector_store = request.app.state.vector_store
    return IngestService(vector_store=vector_store, upload_dir=cfg.UPLOAD_DIR)

@router.get("/upload")
async def get_upload_ui(request: Request):
    """
    Trả về giao diện HTML nhúng vào iframe.
    """
    return templates.TemplateResponse("index.html", {"request": request})
@router.post("/upload")
async def upload_document(request: Request = None, service: IngestService = Depends(get_ingest_service), file: UploadFile = File(...)):
    """
    API Upload tài liệu.
    Mọi logic validate file, check mime type, indexing đều nằm trong Service.
    """
    try:
        return StreamingResponse(
            service.process_upload(file),
            media_type="application/x-ndjson",
        )

    except HTTPException as e:
        # Re-raise để FastAPI trả đúng mã lỗi (400, 500) mà Service đã định nghĩa
        raise e
    except Exception as e:
        # Bắt các lỗi không xác định khác (nếu có sót)
        logger.error(f"Unhandled Error in Upload Route: {e}")
        raise HTTPException(status_code=500, detail="Lỗi không xác định từ máy chủ.")
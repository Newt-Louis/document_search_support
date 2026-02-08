import logging
from fastapi import APIRouter, UploadFile, File, HTTPException, Request, Depends
from app.services.rag.ingest import IngestService

router = APIRouter(tags=["ingest"])
logger = logging.getLogger(__name__)

def get_ingest_service(request: Request) -> IngestService:
    cfg = request.app.state.cfg
    vector_store = request.app.state.vector_store
    return IngestService(vector_store=vector_store, upload_dir=cfg.UPLOAD_DIR)

@router.post("/upload")
async def upload_document(request: Request = None, service: IngestService = Depends(get_ingest_service), file: UploadFile = File(...)):
    """
    API Upload tài liệu.
    Mọi logic validate file, check mime type, indexing đều nằm trong Service.
    """
    if request:
        pass
    try:
        success_message = await service.process_upload(file)

        return {
            "status": "success",
            "filename": file.filename,
            "message": success_message
        }

    except HTTPException as e:
        # Re-raise để FastAPI trả đúng mã lỗi (400, 500) mà Service đã định nghĩa
        raise e
    except Exception as e:
        # Bắt các lỗi không xác định khác (nếu có sót)
        logger.error(f"Unhandled Error in Upload Route: {e}")
        raise HTTPException(status_code=500, detail="Lỗi không xác định từ máy chủ.")
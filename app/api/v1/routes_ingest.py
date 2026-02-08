import os, logging, shutil
from fastapi import APIRouter, UploadFile, File, HTTPException, Request, Depends
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from app.services.rag.ingest import IngestService

router = APIRouter(tags=["ingest"])
logger = logging.getLogger(__name__)

def get_ingest_service(request: Request) -> IngestService:
    cfg = request.app.state.cfg
    vector_store = request.app.state.vector_store
    return IngestService(vector_store=vector_store, upload_dir=cfg.UPLOAD_DIR)

@router.post("/upload")
async def upload_document(request: Request = None, service: IngestService = Depends(get_ingest_service), file: UploadFile = File(...)):
    message = await service.process_upload(file)
    if request:
        pass
    return {
        "status": "success",
        "filename": file.filename,
        "message": "Đã học xong tài liệu mới!"
    }
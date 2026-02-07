import os
import shutil
from fastapi import APIRouter, UploadFile, File, HTTPException, Request
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext

router = APIRouter(tags=["ingest"])

@router.post("/upload")
async def upload_document(request: Request, file: UploadFile = File(...)):
    cfg = request.app.state.cfg

    # 1) Save file
    os.makedirs(cfg.UPLOAD_DIR, exist_ok=True)
    file_path = os.path.join(cfg.UPLOAD_DIR, file.filename)

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 2) Load + index into Qdrant
        documents = SimpleDirectoryReader(input_files=[file_path]).load_data()

        storage_context = StorageContext.from_defaults(vector_store=request.app.state.vector_store)
        index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)

        # 3) Refresh query_engine
        request.app.state.query_engine = index.as_query_engine(streaming=True, similarity_top_k=2)

        return {"status": "success", "filename": file.filename, "message": "Đã học xong tài liệu mới!"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
import logging,sys,os, shutil
import magic
from pathlib import Path
from fastapi import UploadFile, HTTPException
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from app.core.config import get_config

cfg = get_config()
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(Path(cfg.LOG_DIR)/"ingest.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)
CACHE_DIR = cfg.CACHE_DIR
ALLOWED_EXTENSIONS = {
    "application/pdf": ".pdf",
    "text/plain": ".txt",
    "text/csv": ".csv",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
    "application/msword": ".doc",
    "application/vnd.ms-excel": ".xls"
}

class IngestService:
    def __init__(self, vector_store, upload_dir: str):
        self.vector_store = vector_store
        self.upload_dir = upload_dir

    async def _validate_file(self,file:UploadFile):
        header = await file.read(2048)
        mime_type = magic.from_buffer(header, mime=True)
        logger.info(f">>> Phát hiện file có MIME type: {mime_type}")
        await file.seek(0)
        if mime_type not in ALLOWED_EXTENSIONS:
            if not (mime_type.startswith("text/") and file.filename.endswith((".txt", ".csv"))):
                logger.warning(f"File bị reject: {file.filename} (MIME: {mime_type})")
                raise HTTPException(
                    status_code=400,
                    detail=f"File giả mạo hoặc không hỗ trợ! Phát hiện định dạng thực tế: {mime_type}"
                )

    async def process_upload(self, file: UploadFile) -> str:
        try:
            await self._validate_file(file)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        os.makedirs(self.upload_dir, exist_ok=True)
        file_path = os.path.join(self.upload_dir, file.filename)

        try:
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            logger.info(f"File đã lưu: {file_path}")
            return self.index_file(file_path)

        except Exception as e:
            logger.exception(f"Lỗi khi xử lý file {file.filename}")
            if os.path.exists(file_path):
                os.remove(file_path)
            raise e

    def index_file(self, file_path: str)->str:
        try:
            documents = SimpleDirectoryReader(input_files=[file_path]).load_data()
            storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
            VectorStoreIndex.from_documents(
                documents,
                storage_context=storage_context,
                show_progress=True
            )
            return f"Lập chỉ mục cho tài liệu thành công tại: {os.path.basename(file_path)}"
        except Exception as e:
            if os.path.exists(file_path):
                os.remove(file_path)
            raise HTTPException(status_code=500, detail=f"Lỗi khi Indexing: {str(e)}")
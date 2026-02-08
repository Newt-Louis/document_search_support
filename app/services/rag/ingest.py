import logging,sys,os, shutil
from typing import List
from fastapi import UploadFile, HTTPException
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, Settings
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.llms.ollama import Ollama
import qdrant_client

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
CACHE_DIR = os.getenv("CACHE_DIR", "../../data/cache")
ALLOWED_EXTENSIONS = {".txt", ".csv", ".docx", ".pdf", ".xlsx"}

class IngestService:
    def __init__(self, vector_store, upload_dir: str):
        self.vector_store = vector_store
        self.upload_dir = upload_dir

    @staticmethod
    def _validate_file(filename: str):
        ext = os.path.splitext(filename)[1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise ValueError(f"Định dạng file '{ext}' không được hỗ trợ. Chỉ chấp nhận: {', '.join(ALLOWED_EXTENSIONS)}")

    async def process_upload(self, file: UploadFile) -> str:
        try:
            self._validate_file(file.filename)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        os.makedirs(self.upload_dir, exist_ok=True)
        file_path = os.path.join(self.upload_dir, file.filename)

        try:
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            return self.index_file(file_path)

        except Exception as e:
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
                    raise HTTPException(status_code=500, detail=f"Lỗi khi Indexing: {str(e)}")
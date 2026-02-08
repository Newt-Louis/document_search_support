import logging, sys, os, shutil, magic, csv, openpyxl
from typing import Generator, List
from pathlib import Path
from fastapi import UploadFile, HTTPException

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, Settings, Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline
from llama_index.vector_stores.qdrant import QdrantVectorStore

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
    def __init__(self, vector_store: QdrantVectorStore, upload_dir: str):
        self.vector_store = vector_store
        self.upload_dir = upload_dir
        self.text_splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=200)

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

            success = await self.index_file(file_path)
            if success:
                return f"Lập chỉ mục cho tài liệu {file.filename} thành công!"
            return "Có lỗi khác exception không bắt được !!!"

        except Exception as e:
            logger.exception(f"Lỗi khi xử lý file {file.filename}")
            if os.path.exists(file_path):
                os.remove(file_path)
            raise e

    async def index_file(self, file_path: str)->bool:
        try:
            # documents = SimpleDirectoryReader(input_files=[file_path]).load_data()
            # storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
            pipeline = IngestionPipeline(
                transformations=[
                    self.text_splitter,  # 1. Cắt nhỏ (Chunking)
                    Settings.embed_model,  # 2. Hóa vector (Embedding)
                ],
                vector_store=self.vector_store,  # 3. Đẩy vào Qdrant
            )
            total_chunks = 0
            for doc_batch in self._lazy_load_file(file_path, chunk_size_mb=40):
                if not doc_batch:
                    continue

                # Chạy pipline cho 1 batch nhỏ
                # Dữ liệu vào -> Cắt -> Embed -> Lưu Qdrant -> Xóa khỏi RAM
                nodes = await pipeline.arun(documents=doc_batch, show_progress=False)
                total_chunks += len(nodes)
                logger.info(f">>> Đã đẩy xong {len(nodes)} chunks vào DB...")

                # Force Python dọn rác ngay lập tức (Optional nhưng tốt cho 20GB)
                del doc_batch
                del nodes
            # VectorStoreIndex.from_documents(
            #     documents,
            #     storage_context=storage_context,
            #     show_progress=True
            # )
            return True
        except Exception as e:
            logger.exception(f"Lỗi khi lập chỉ mục: {e}")
            if os.path.exists(file_path):
                os.remove(file_path)
            raise HTTPException(status_code=500, detail=f"Lỗi khi Indexing: {str(e)}")

    def _lazy_load_file(self, file_path: str, chunk_size_mb: int = 10) -> Generator[List[Document], None, None]:
        """
        Đọc file theo từng phần nhỏ (Batch).
        chunk_size_mb: Kích thước mỗi lần đọc (Mặc định 10MB text)
        """
        ext = os.path.splitext(file_path)[1].lower()

        if ext == ".txt":
            # Đọc file Text dòng theo dòng
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                batch_text = ""
                current_size = 0

                for line in f:
                    batch_text += line
                    current_size += len(line.encode('utf-8'))

                    # Khi đủ 10MB thì yield ra một Document rồi xóa RAM biến batch_text
                    if current_size >= chunk_size_mb * 1024 * 1024:
                        yield [Document(text=batch_text, metadata={"filename": os.path.basename(file_path)})]
                        batch_text = ""  # Reset RAM
                        current_size = 0

                # Yield nốt phần còn dư cuối cùng
                if batch_text:
                    yield [Document(text=batch_text, metadata={"filename": os.path.basename(file_path)})]

        elif ext == ".csv":
            # Đọc CSV dòng theo dòng
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                reader = csv.reader(f)
                batch_text = ""
                current_size = 0
                for row in reader:
                    row_text = ", ".join(row) + "\n"
                    batch_text += row_text
                    current_size += len(row_text.encode('utf-8'))

                    if current_size >= chunk_size_mb * 1024 * 1024:
                        yield [Document(text=batch_text, metadata={"filename": os.path.basename(file_path)})]
                        batch_text = ""
                        current_size = 0
                if batch_text:
                    yield [Document(text=batch_text, metadata={"filename": os.path.basename(file_path)})]

        elif ext == ".pdf":
            # Với PDF, ta dùng pypdf đọc từng trang
            import pypdf
            reader = pypdf.PdfReader(file_path)
            for page in reader.pages:
                text = page.extract_text()
                # Yield từng trang một. Mỗi trang là 1 Document riêng.
                if text:
                    yield [Document(text=text, metadata={"page_label": reader.get_page_number(page),
                                                         "filename": os.path.basename(file_path)})]

        elif ext == ".xlsx":
            # Quan trọng: read_only=True giúp openpyxl không load hết vào RAM
            # data_only=True để lấy giá trị cuối cùng, không lấy công thức hàm
            wb = openpyxl.load_workbook(file_path, read_only=True, data_only=True)

            for sheet in wb:
                batch_text = ""
                current_size = 0
                # Duyệt từng dòng trong sheet. Đây là Generator.
                for row in sheet.iter_rows(values_only=True):
                    # Lọc None và chuyển thành string, cách nhau bởi dấu phẩy
                    row_values = [str(cell) if cell is not None else "" for cell in row]
                    row_text = ", ".join(row_values) + "\n"

                    batch_text += row_text
                    current_size += len(row_text.encode('utf-8'))

                    # Kiểm tra ngưỡng chunk_size_mb
                    if current_size >= chunk_size_mb * 1024 * 1024:
                        yield [Document(text=batch_text,
                                        metadata={"filename": os.path.basename(file_path), "sheet": sheet.title})]
                        batch_text = ""  # Giải phóng RAM
                        current_size = 0

                # Yield phần còn dư của sheet hiện tại
                if batch_text:
                    yield [Document(text=batch_text,
                                    metadata={"filename": os.path.basename(file_path), "sheet": sheet.title})]
            wb.close()

        elif ext == ".docx":
            from docx import Document as DocxDocument
            # Word file là tập hợp các đoạn văn (Paragraphs)
            doc = DocxDocument(file_path)
            batch_text = ""
            current_size = 0

            for para in doc.paragraphs:
                text = para.text
                if not text.strip():
                    continue  # Bỏ qua đoạn trống

                text += "\n"
                batch_text += text
                current_size += len(text.encode('utf-8'))

                # Kiểm tra ngưỡng chunk_size_mb
                if current_size >= chunk_size_mb * 1024 * 1024:
                    yield [Document(text=batch_text, metadata={"filename": os.path.basename(file_path)})]
                    batch_text = ""
                    current_size = 0

            # Yield nốt phần còn dư
            if batch_text:
                yield [Document(text=batch_text, metadata={"filename": os.path.basename(file_path)})]
        else:
            yield []
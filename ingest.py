# ingest.py
import logging
import sys
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, Settings
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
import qdrant_client

# 1. Cấu hình Log để dễ debug
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# 2. Setup Global Settings (Cấu hình toàn cục)
# KHÔNG dùng OpenAI, chỉ dùng đồ nhà trồng được (Local)
print(">>> Đang cấu hình kết nối tới Ollama...")

# Model nhúng (Embedding) - Chịu trách nhiệm biến chữ thành số
Settings.embed_model = OllamaEmbedding(
    model_name="bge-m3",
    base_url="http://localhost:11434",
    ollama_additional_kwargs={"mirostat": 0}
)

# Model ngôn ngữ (LLM) - Dùng để tóm tắt hoặc xử lý metadata nếu cần
Settings.llm = Ollama(
    model="llama3.2",
    request_timeout=360.0
)

# 3. Kết nối tới Qdrant (Vector DB)
print(">>> Đang kết nối tới Qdrant Database...")
client = qdrant_client.QdrantClient(
    # Docker của ta đang chạy ở port 6333
    url="http://localhost:6333"
)

# Tạo một "Collection" (giống như Table trong SQL) tên là "company_docs"
vector_store = QdrantVectorStore(client=client, collection_name="company_docs")
storage_context = StorageContext.from_defaults(vector_store=vector_store)


# 4. Đọc dữ liệu và Indexing (Phần nặng nhất)
def ingest_data():
    print(">>> Đang đọc file từ thư mục 'data/'. Quá trình này có thể mất vài giây...")

    # Đọc tất cả file trong folder data
    documents = SimpleDirectoryReader("./data").load_data()

    # MAGIC HAPPENS HERE:
    # 1. Cắt nhỏ văn bản (Chunking)
    # 2. Gọi bge-m3 để biến thành Vector
    # 3. Đẩy Vector vào Qdrant
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
    )

    print(">>> ✅ Xong! Dữ liệu đã được nạp vào Qdrant.")
    return index


if __name__ == "__main__":
    ingest_data()
# query.py
import logging
import sys
from llama_index.core import VectorStoreIndex, Settings
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
import qdrant_client
from llama_index.core.prompts import PromptTemplate

# 1. Cấu hình y hệt như lúc Ingest (để đảm bảo đồng bộ vector)
Settings.embed_model = OllamaEmbedding(
    model_name="bge-m3",
    base_url="http://localhost:11434"
)

# Cấu hình LLM: Llama-3.2 chạy local
Settings.llm = Ollama(
    model="llama3.2",
    request_timeout=360.0,
    temperature=0.1  # Giữ nhiệt độ thấp để model trả lời trung thực, ít sáng tạo linh tinh
)

# 2. Kết nối lại vào Qdrant (Chỉ connect, không tạo mới)
client = qdrant_client.QdrantClient(url="http://localhost:6333")
vector_store = QdrantVectorStore(client=client, collection_name="company_docs")

# 3. Load Index từ Vector DB lên (Siêu nhẹ, không tốn RAM load data gốc)
index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

# 4. Cấu hình Prompt "Thần thánh" (System Prompt)
# Đây là chỗ ta dạy model nói chuyện theo phong cách bạn muốn
qa_prompt_str = (
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
qa_template = PromptTemplate(qa_prompt_str)


def chat_loop():
    print("\n>>> 🤖 System Ready! Gõ 'exit' để thoát.")

    # Tạo Query Engine (Bộ máy truy vấn)
    # similarity_top_k=3 nghĩa là chỉ lấy 3 đoạn văn bản giống nhất để gửi cho AI
    query_engine = index.as_query_engine(
        text_qa_template=qa_template,
        similarity_top_k=3
    )

    while True:
        user_input = input("\nBạn: ")
        if user_input.lower() in ["exit", "quit"]:
            break

        # MAGIC HAPPENS HERE:
        response = query_engine.query(user_input)

        print(f"\nSyezain AI: {response}")


if __name__ == "__main__":
    chat_loop()
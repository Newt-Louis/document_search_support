# app/services/rag/prompts.py
from llama_index.core.prompts import PromptTemplate

QA_PROMPT_STR = (
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

QA_TEMPLATE = PromptTemplate(QA_PROMPT_STR)

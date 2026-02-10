from sentence_transformers import SentenceTransformer
import os

# Đổi model sang E5-Large
MODEL_NAME = "intfloat/multilingual-e5-small"
# CACHE_DIR = "./data/cache/multilingual-e5-large"
CACHE_DIR = os.getenv("CACHE_DIR", "./data/cache") + "/multilingual-e5-small"

print(f"⏳ Đang tải model {MODEL_NAME} về thư mục: {os.path.abspath(CACHE_DIR)}...")

# Model này nằm trong danh sách hỗ trợ, nên sẽ chạy mượt
model = SentenceTransformer(MODEL_NAME, cache_folder=CACHE_DIR)

print(f"✅ Đã tải xong! Kiểm tra thư mục '{CACHE_DIR}' để thấy file.")
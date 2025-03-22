import os
import torch

# Base directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")

# Directory paths
LOGS_DIR = os.path.join(DATA_DIR, "logs")
DOCUMENTS_DIR = os.path.join(DATA_DIR, "documents")
IMAGES_DIR = os.path.join(DATA_DIR, "images")
VECTORSTORE_DIR = os.path.join(DATA_DIR, "vectorstore")
DATABASE_DIR = os.path.join(DATA_DIR, "database")

# File paths
DATABASE_PATH = os.path.join(DATABASE_DIR, "threads.db")
LOG_FILE = os.path.join(LOGS_DIR, "app.log")
JSON_LOG_FILE = os.path.join(LOGS_DIR, "analytics.json")

# API settings
OLLAMA_API_BASE = os.environ.get("OLLAMA_API_BASE", "http://localhost:11434")

# Model settings
DEFAULT_MODEL = "deepseek-r1:8b"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Retrieval settings
RETRIEVER_K = 4  # Number of documents to retrieve

# Multimodal model settings
MULTIMODAL_MODELS = [
    "gemma:2b-it",
    "llava:7b",
    "llava:13b",
    "llava:34b",
    "bakllava:7b",
    "moondream:7b",
    "gemma3:12b"
]

# Supported image types
SUPPORTED_IMAGE_TYPES = ["jpg", "jpeg", "png", "gif"]

# Thread settings
DEFAULT_THREAD_NAME = "New Chat"

# Logging settings
LOG_LEVEL = "INFO"
MAX_LOG_FILES = 10
MAX_LOG_SIZE_MB = 10

# Create necessary directories
for directory in [LOGS_DIR, DOCUMENTS_DIR, IMAGES_DIR, VECTORSTORE_DIR, DATABASE_DIR]:
    os.makedirs(directory, exist_ok=True) 
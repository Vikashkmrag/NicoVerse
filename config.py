"""
Configuration module for the Document Retrieval application.
This file imports and re-exports all settings from modules/utils/config.py.
"""

# Import all settings from the utils config module
from modules.utils.config import *

"""
Configuration settings for the Document Retrieval Application.
This file centralizes all configurable parameters to make the app more maintainable.
"""

import os
import torch

# Base directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

# Directory paths
LOGS_DIR = os.path.join(DATA_DIR, "logs")
DOCUMENTS_DIR = os.path.join(DATA_DIR, "documents")
IMAGES_DIR = os.path.join(DATA_DIR, "images")
VECTORSTORE_DIR = os.path.join(DATA_DIR, "vectorstore")
DATABASE_DIR = os.path.join(DATA_DIR, "database")
BACKUP_DIR = os.path.join(DATA_DIR, "backups")

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
FALLBACK_EMBEDDING_MODEL = "nomic-embed-text"  # Fallback model for embeddings if primary fails

# Document processing settings
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
SUPPORTED_FILE_TYPES = ["pdf", "txt"]

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
DEFAULT_MULTIMODAL_MODEL = "gemma:2b-it"  # Default model for image processing

# Supported image types
SUPPORTED_IMAGE_TYPES = ["jpg", "jpeg", "png", "gif"]
SUPPORTED_TEXT_TYPES = ["pdf", "txt"]
# UI settings
APP_TITLE = "Document Retrieval App"
APP_LAYOUT = "wide"
SIDEBAR_TITLE = "Document Retrieval"

# Thread settings
DEFAULT_THREAD_NAME = "New Chat"
DEFAULT_THREAD_FORMAT = "Thread %Y-%m-%d %H:%M"  # strftime format

# Logging settings
LOG_LEVEL = "INFO"
MAX_LOG_FILES = 10
MAX_LOG_SIZE_MB = 10

# Debug settings
DEBUG_MODE = False  # Set to False to disable debug prints
DEBUG_FORMAT = "[DEBUG][{file}:{line}] {message}"  # Format for debug messages

# API timeouts (in seconds)
FAST_TIMEOUT = 1
STANDARD_TIMEOUT = 5
LONG_TIMEOUT = 10

# Create necessary directories
for directory in [LOGS_DIR, DOCUMENTS_DIR, IMAGES_DIR, VECTORSTORE_DIR, DATABASE_DIR, BACKUP_DIR]:
    os.makedirs(directory, exist_ok=True) 
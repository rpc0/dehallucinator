"""Configuration settings for Application API"""

import os
from dotenv import load_dotenv

load_dotenv()

GPU_ENABLED: bool = os.environ.get("GPU_ENABLED", "true").lower() == "true"

DATA_FOLDER = os.environ.get("DATA_FOLDER", "/data/contexts")
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://172.17.0.1:7869")
CHROMA_DB_HOST = os.environ.get("CHROMA_DB_HOST", "chromadb")
CHROMA_DB_COLLECTION = os.environ.get("CHROMA_DB_COLLECTION", "Squad2.0")

LLM_MODEL = os.environ.get("LLM_MODEL", "llama3.1:8b-instruct-q3_K_L")
LLM_PROMPT = os.environ.get(
    "LLM_PROMPT", "rlm/rag-prompt-llama"
)  # https://smith.langchain.com/hub/rlm/rag-prompt-llama

EMBEDDING_MODEL = os.environ.get(
    "EMBEDDING_MODEL", "sentence-transformers/all-mpnet-base-v2"
)

SIMILARITY_THRESHOLD: float = float(os.environ.get("SIMILARITY_THRESHOLD", "1.0"))
CONTEXT_DOCUMENTS_RETRIEVED: int = int(
    os.environ.get("CONTEXT_DOCUMENTS_RETRIEVED", "6")
)

TXT_CHUNK_SIZE: int = int(os.environ.get("TXT_CHUNK_SIZE", "1500"))
TXT_CHUNK_OVERLAP: int = int(os.environ.get("TXT_CHUNK_OVERLAP", "100"))

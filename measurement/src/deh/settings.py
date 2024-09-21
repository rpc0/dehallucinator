"""Configuration settings for Assessment Models."""

import os
from dotenv import load_dotenv

load_dotenv()

API_ANSWER_ENDPOINT = os.environ.get("API_ANSWER_ENDPOINT", "localhost/api/")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:7869")
ASSESSMENT_LLM_MODEL = os.environ.get("ASSESSMENT_LLM_MODEL", "tinyllama:latest")
ASSESSMENT_EMBEDDING_MODEL = os.environ.get(
    "ASSESSMENT_EMBEDDING_MODEL", "all-minilm:latest"
)

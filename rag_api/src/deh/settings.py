"""Configuration settings for Application API"""

import os
from dotenv import load_dotenv

load_dotenv()

DATA_FOLDER = os.environ.get("DATA_FOLDER", "/data")
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://172.17.0.1:7869")
LLM_MODEL = os.environ.get("LLM_MODEL", "llava-phi3:latest")

import json
import os
import time

import numpy as np
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

app = FastAPI()

# Enable CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_credentials=False,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Application variables

vectorstore = None


@app.on_event("startup")
async def startup_event():
    """Cache vector store at start-up."""
    try:
        initialize_vectorstore()
    except Exception as exc:
        print(str(exc))


def initialize_vectorstore():
    """Initialize and cache vector store interface."""

    global vectorstore

    # Initialize model from artifact-store:
    if not vectorstore:
        # TODO: Initialize vector store
        vectorstore = True


@app.get("/")
async def root():
    return {"message": "DEH Application APIs"}


@app.get("/doc/load")
async def load_model(doc_path: str, doc_type: str):
    """Load documents into vector store."""

    global vectorstore

    # TODO: Implement doc count
    return {"status": "success", "doc_path": doc_path, "doc_count": 0}


@app.post("/respond")
async def respond(query: str):
    """Provides an LLM response based on query."""
    # https://towardsdatascience.com/building-a-rag-chain-using-langchain-expression-language-lcel-3688260cad05
    # TODO: Implement
    return {"response": ""}

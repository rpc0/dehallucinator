from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain import hub
from langchain_community.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler


import logging

import deh.settings as settings
from deh.utils import format_context_documents as format_docs

app = FastAPI()

# Enable CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_credentials=False,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

logger = logging.getLogger(__name__)

# Global Application variables
vector_store = None
document_location = settings.DATA_FOLDER


@app.on_event("startup")
async def startup_event():
    """Cache vector store at start-up."""
    try:
        initialize_vectorstore(document_location, "**/*.context")

    except Exception as exc:
        print(str(exc))


def initialize_vectorstore(doc_loc: str, doc_filter="**/*.*"):
    """Initialize and cache vector store interface."""

    global vector_store

    # Initialize model from artifact-store:
    if not vector_store:
        loader = DirectoryLoader(path=doc_loc, glob=doc_filter, silent_errors=True)
        data = loader.load()
        doc_count = len(data)
        logger.debug(f"Loaded {doc_count} documents")

        # Split into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500, chunk_overlap=100
        )
        all_splits = text_splitter.split_documents(data)
        logger.debug(f"Split into {len(all_splits)} chunks")

        vector_store = Chroma.from_documents(
            documents=all_splits,
            embedding=HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-mpnet-base-v2"
            ),
        )

        return doc_count


@app.get("/")
async def root():
    """Hello world default end-point for uptime checking."""
    return {"message": "DEH Application APIs"}


@app.get("/doc/load")
async def load_model(doc_path: str, doc_filter: str):
    """Re-initializes vector store with new document corpus."""

    global vector_store
    vector_store = None

    doc_count = initialize_vectorstore(doc_path, doc_filter)

    return {"status": "success", "doc_path": doc_path, "doc_count": doc_count}


@app.get("/answer")
async def answer(question: str):
    """Provides an LLM response based on query."""
    # https://towardsdatascience.com/building-a-rag-chain-using-langchain-expression-language-lcel-3688260cad05

    retriever = vector_store.as_retriever()
    response = basic_rag_chain(retriever, question)
    return {"response": response}


def basic_rag_chain(retriever, question):
    """Simplest RAG Chain (v1) implementation."""

    qa_prompt = hub.pull("rlm/rag-prompt-llama")
    llm = Ollama(
        base_url=settings.OLLAMA_HOST,
        model=settings.LLM_MODEL,
        verbose=True,
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
    )

    rag_chain = (
        RunnableParallel(
            context=retriever | format_docs, question=RunnablePassthrough()
        )
        | qa_prompt
        | llm
    )
    return rag_chain.invoke(question)

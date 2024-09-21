from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain import hub
from langchain_community.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from operator import itemgetter

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

# Application Status Parameters:
TXT_SPLITTER = ""
DOCS_LOADED = 0


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
    global TXT_SPLITTER
    global DOCS_LOADED

    # Initialize model from artifact-store:
    if not vector_store:
        text_loader_kwargs = {"autodetect_encoding": True}

        loader = DirectoryLoader(
            path=doc_loc,
            glob=doc_filter,
            loader_cls=TextLoader,
            loader_kwargs=text_loader_kwargs,
            silent_errors=True,
        )
        data = loader.load()
        doc_count = len(data)
        logger.info(f"Loaded {doc_count} documents")

        # Split into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.TXT_CHUNK_SIZE, chunk_overlap=settings.TXT_CHUNK_OVERLAP
        )

        # Store text splitter class type for logging:
        TXT_SPLITTER = text_splitter.__class__.__name__

        all_splits = text_splitter.split_documents(data)
        logger.debug(f"Split into {len(all_splits)} chunks")

        vector_store = Chroma.from_documents(
            documents=all_splits,
            embedding=HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL),
        )

        DOCS_LOADED = doc_count
        return doc_count


@app.get("/")
async def root():
    """Hello world default end-point for application key parameter visibilty."""
    return {
        "application": "DEH Application APIs",
        # Model and Vector Store Params:
        "llm_model": settings.LLM_MODEL,
        "llm_prompt": settings.LLM_PROMPT,
        "embedding_model": settings.EMBEDDING_MODEL,
        "text_splitter": TXT_SPLITTER,
        "text_chunk_size": settings.TXT_CHUNK_SIZE,
        "text_chunk_overlap": settings.TXT_CHUNK_OVERLAP,
        # Doc Count:
        "docs_loaded": DOCS_LOADED,
    }


@app.get("/doc/load")
async def load_model(doc_path: str, doc_filter: str):
    """Re-initializes vector store with new document corpus."""

    # TODO: Need to further implement/check (future feature)
    global vector_store
    vector_store = None

    doc_count = initialize_vectorstore(doc_path, doc_filter)

    return {"status": "success", "doc_path": doc_path, "doc_count": doc_count}


@app.get("/answer")
async def answer(question: str):
    """Provides an LLM response based on query."""
    # https://towardsdatascience.com/building-a-rag-chain-using-langchain-expression-language-lcel-3688260cad05

    retriever = vector_store.as_retriever()

    llm = Ollama(
        base_url=settings.OLLAMA_HOST,
        model=settings.LLM_MODEL,
        verbose=True,
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
    )

    response = basic_rag_chain_with_context(retriever, question, llm)
    return {
        "response": response,
        # Diagnostic values used for measurement logging, etc:
        "llm_model": settings.LLM_MODEL,
        "llm_prompt": settings.LLM_PROMPT,
        "embedding_model": settings.EMBEDDING_MODEL,
        "text_splitter": TXT_SPLITTER,
        "text_chunk_size": settings.TXT_CHUNK_SIZE,
        "text_chunk_overlap": settings.TXT_CHUNK_OVERLAP,
    }


def basic_rag_chain(retriever, question, llm):
    """Simplest RAG Chain (v0) implementation."""

    qa_prompt = hub.pull("rlm/rag-prompt-llama")

    rag_chain = (
        RunnableParallel(
            context=retriever | format_docs, question=RunnablePassthrough()
        )
        | qa_prompt
        | llm
    )
    return rag_chain.invoke(question)


def basic_rag_chain_with_context(retriever, question, llm):
    """Simplest RAG Chain (v0.1) implementation which includes context pass-through."""

    qa_prompt = hub.pull(settings.LLM_PROMPT)

    #    rag_chain = RunnableParallel(
    #        context=retriever | format_docs, question=RunnablePassthrough()
    #    ) | RunnableParallel(qa_prompt | llm, context=itemgetter("context"))

    rag_chain = RunnableParallel(
        context=retriever | format_docs, question=RunnablePassthrough()
    ) | RunnableParallel(
        answer=qa_prompt | llm,
        question=itemgetter("question"),
        context=itemgetter("context"),
    )

    return rag_chain.invoke(question)

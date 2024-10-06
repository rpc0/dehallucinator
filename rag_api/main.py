from fastapi import FastAPI

from starlette.middleware.cors import CORSMiddleware

import chromadb

from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from contextlib import asynccontextmanager
from operator import itemgetter
import logging
from tqdm import tqdm

import deh.settings as settings
import deh.guardrail as guardrail
from deh.utils import format_context_documents as format_docs
from deh.utils import retriever_with_scores, dedupulicate_contexts
from deh.prompts import (
    qa_eval_prompt_with_context_text,
    LLMEvalResult,
    rag_prompt_llama_text,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)  # logging.basicConfig(level=logging.DEBUG)


# Global Application variables
VECTOR_STORE = None
TXT_SPLITTER = ""
DOCS_LOADED = 0


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Responsible for managing start-up and shutdown.
    https://fastapi.tiangolo.com/advanced/events/#lifespan
    """

    # Initialize vector store:
    initialize_vectorstore(settings.DATA_FOLDER, "**/*.context")
    yield


# Create the FASTAPI App
app = FastAPI(lifespan=lifespan)

# Enable CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_credentials=False,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def initialize_vectorstore(doc_loc: str, doc_filter="**/*.*"):
    """Initialize and cache vector store interface."""

    logger.info("Initializing vector store...")
    global VECTOR_STORE
    global TXT_SPLITTER
    global DOCS_LOADED

    # Initialize remote connection to ChromaDB vector store:
    remote_chroma_client = chromadb.HttpClient(host=settings.CHROMA_DB_HOST, port=8000)

    collection_exists = False
    try:
        collection = remote_chroma_client.get_collection(settings.CHROMA_DB_COLLECTION)
        DOCS_LOADED = collection.count()
        collection_exists = True
        logger.info("Collection already exists.")
    except:
        logger.info("Collection does not exists, needs to be created.")
        remote_chroma_client.create_collection(name=settings.CHROMA_DB_COLLECTION)

    # Initialize embedding function:
    embedding_fxn = HuggingFaceEmbeddings(
        model_name=settings.EMBEDDING_MODEL,
        # model_name="../data/model_cache",
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True},
    )

    VECTOR_STORE = Chroma(
        collection_name=settings.CHROMA_DB_COLLECTION,
        embedding_function=embedding_fxn,
        client=remote_chroma_client,
    )
    logger.info("Vector store created.")

    # If collection didn't exist, load documents and embeddings:
    if not collection_exists:

        loader = DirectoryLoader(
            path=doc_loc,
            glob=doc_filter,
            loader_cls=TextLoader,
            loader_kwargs={"autodetect_encoding": True},
            silent_errors=False,
        )

        data = list(tqdm(loader.load(), desc="Loading documents"))

        DOCS_LOADED = len(data)
        logger.info(f"Loaded {DOCS_LOADED} documents")

        # Split into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.TXT_CHUNK_SIZE, chunk_overlap=settings.TXT_CHUNK_OVERLAP
        )

        # Store text splitter class type for logging:
        TXT_SPLITTER = text_splitter.__class__.__name__

        all_splits = text_splitter.split_documents(data)
        logger.info(f"Split into {len(all_splits)} chunks")

        # Load documents:
        # https://docs.trychroma.com/guides
        VECTOR_STORE.add_documents(all_splits)

        return DOCS_LOADED


def get_settings():
    """JSON encoding of system settings."""
    return {
        # Model and Vector Store Params:
        "llm_model": settings.LLM_MODEL,
        "llm_prompt": settings.LLM_PROMPT,
        "embedding_model": settings.EMBEDDING_MODEL,
        "text_splitter": TXT_SPLITTER,
        "text_chunk_size": settings.TXT_CHUNK_SIZE,
        "text_chunk_overlap": settings.TXT_CHUNK_OVERLAP,
        "context_similarity_threshold": settings.SIMILARITY_THRESHOLD,
        "context_docs_retrieved": settings.CONTEXT_DOCUMENTS_RETRIEVED,
        # Doc Count:
        "docs_loaded": DOCS_LOADED,
    }


@app.get("/")
async def root():
    """Hello world default end-point for application key parameter visibilty."""
    return {"application": "DEH Application APIs", "system_settings": get_settings()}


@app.get("/doc/load")
async def load_model(doc_path: str, doc_filter: str):
    """Re-initializes vector store with new document corpus."""

    global VECTOR_STORE
    vector_store = None
    initialize_vectorstore(doc_path, doc_filter)

    return {"status": "success", "doc_path": doc_path, "doc_count": DOCS_LOADED}


@app.get("/answer")
async def answer(question: str):
    """Provides an LLM response based on query."""
    # https://towardsdatascience.com/building-a-rag-chain-using-langchain-expression-language-lcel-3688260cad05

    llm = Ollama(base_url=settings.OLLAMA_HOST, model=settings.LLM_MODEL, verbose=True)

    try:
        response = rag_chain(question, llm)
    except guardrail.GuardRailException as exc:
        logger.info("GuardRailException: " + ",".join(exc.args))
        response = {"errors": exc.args, "system_settings": get_settings()}

    return {
        # RAG chain response:
        "response": response,
        # System values used for measurement logging, etc:
        "system_settings": get_settings(),
    }


def rag_chain(question, llm):
    """RAG Chain implementation including:
    - Context similarity guardrail
    - LLM as judge evaluation

    Results in dictionary:
    - evaluation - llm evaluation and rationale
    - answer - llm generated response
    - question - original query
    - context - array of context docs & meta data
    """

    # Structure definition for evaluation result:
    json_parser = JsonOutputParser(pydantic_object=LLMEvalResult)

    # Initial LLM generation prompt:
    qa_prompt = PromptTemplate(
        template=rag_prompt_llama_text, input_variables=["question", "context"]
    )

    # LLM-as-judge evaluation prompt:
    qa_eval_prompt_with_context = PromptTemplate(
        template=qa_eval_prompt_with_context_text,
        input_variables=["question", "answer", "context"],
        partial_variables={
            "format_instructions": json_parser.get_format_instructions()
        },
    )

    # Chain assembly:
    # fmt: off
    rag_chain = (
        # Context retrieval w/ Similarity GuardRail
        RunnableParallel(
            question = RunnablePassthrough(),
            context = retriever_with_scores(VECTOR_STORE) | guardrail.similarity_guardrail(settings.SIMILARITY_THRESHOLD) | dedupulicate_contexts
        )
        | RunnableParallel (
            context = format_docs,
            docs = itemgetter("context"),
            question = itemgetter("question")
        )
        # LLM response generation
        | RunnableParallel(
            question = itemgetter("question"),
            answer= qa_prompt | llm, 
            docs = itemgetter("docs")
        )
        # LLM evaluation
        #| RunnableParallel(
        #    evaluation = qa_eval_prompt_with_context | llm | json_parser,
        #    answer = itemgetter("answer"),
        #    question = itemgetter("question"),
        #    context = itemgetter("docs")
        #)
    )
    # fmt: on

    return rag_chain.invoke(question)

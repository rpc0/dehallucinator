from httpx import request
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
import time, requests, asyncio
from functools import wraps
from pydantic import BaseModel
from typing import Optional

import deh.settings as settings
import deh.guardrail as guardrail
from deh.utils import format_context_documents as format_docs
from deh.utils import retriever_with_scores, dedupulicate_contexts
from deh.prompts import (
    qa_eval_prompt_with_context_text,
    LLMEvalResult,
    rag_prompt_llama_text,
    hyde_prompt_text,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)  # logging.basicConfig(level=logging.DEBUG)


# Global Application variables
VECTOR_STORE = None
LLM = None
TXT_SPLITTER = ""
DOCS_LOADED = 0

# General utility functions:


def getLLM():
    """Returns global LLM for the process."""
    global LLM
    if LLM is None:
        LLM = Ollama(
            base_url=settings.OLLAMA_HOST, model=settings.LLM_MODEL, verbose=True
        )

    return LLM


async def loadLLM():
    """Load LLM model."""
    global LLM
    if LLM is None:
        # Load the LLM onto the Ollama server
        try:
            response = await getLLM().apredict("Hello")
            logger.info(f"LLM loaded onto Ollama server. Response: {response}")
        except Exception as e:
            logger.info(f"Failed to load model onto Ollama server. Error: {e}")
    return None


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


def api_response(response: str) -> str:
    """Utility function to provide consistent API response structure."""
    return {"response": response, "system_settings": get_settings()}


def guardrail_api(api_endpoint):
    """Decorator to catch GuardRail Exceptoins"""

    @wraps(api_endpoint)
    async def wrapper(*args, **kwargs):
        try:
            return await api_endpoint(*args, **kwargs)
        except guardrail.GuardRailException as gexc:
            logger.info("GuardRailException: " + ",".join(gexc.args))
            response = {"errors": gexc.args, "system_settings": get_settings()}
            return response

    return wrapper


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Responsible for managing start-up and shutdown.
    https://fastapi.tiangolo.com/advanced/events/#lifespan
    """
    await asyncio.gather(
        loadLLM(),  # Load LLM model
        initialize_vectorstore(
            settings.DATA_FOLDER, "**/*.context"
        ),  # Initialize vector store
    )
    yield


async def initialize_vectorstore(doc_loc: str, doc_filter="**/*.*"):
    """Initialize and cache vector store interface."""

    logger.info("Initializing vector store...")
    global VECTOR_STORE
    global TXT_SPLITTER
    global DOCS_LOADED

    # Initialize remote connection to ChromaDB vector store:
    remote_chroma_client = chromadb.HttpClient(host=settings.CHROMA_DB_HOST, port=8000)
    heartbeat_confirmation = None
    while heartbeat_confirmation is not None:
        try:
            heartbeat_confirmation = remote_chroma_client.heartbeat()
        except Exception as exc:
            logger.error("ChromaDB heartbeat was not returned.  Waiting to try again.")
            time.sleep(2.5)

    collection_exists = False
    try:
        collection = remote_chroma_client.get_collection(settings.CHROMA_DB_COLLECTION)
        DOCS_LOADED = collection.count()
        collection_exists = True
        logger.info("Collection already exists.")

        # GUARD-Statement: If collection is empty reload documents:
        if DOCS_LOADED == 0:
            collection_exists = False
            logger.info(
                f"Collection {settings.CHROMA_DB_COLLECTION} is empty so needs to be reloaded."
            )

    except:
        logger.info("Collection does not exists, needs to be created.")
        remote_chroma_client.create_collection(
            name=settings.CHROMA_DB_COLLECTION, metadata={"hnsw:space": "cosine"}
        )

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

        if DOCS_LOADED > 0:
            # Split into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=settings.TXT_CHUNK_SIZE,
                chunk_overlap=settings.TXT_CHUNK_OVERLAP,
            )

            # Store text splitter class type for logging:
            TXT_SPLITTER = text_splitter.__class__.__name__

            all_splits = text_splitter.split_documents(data)
            logger.info(f"Split into {len(all_splits)} chunks")

            # Load documents:
            # https://docs.trychroma.com/guides
            VECTOR_STORE.add_documents(all_splits)

        return DOCS_LOADED


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


@app.get("/")
@guardrail_api
async def root():
    """Hello world default end-point for application key parameter visibilty."""
    return {"application": "DEH Application APIs", "system_settings": get_settings()}


@app.get("/error")
@guardrail_api
async def error():
    """Generates exception as part of error checking."""
    raise guardrail.GuardRailException("Testing GuardRail exception.")


@app.get("/hyde")
async def hyde(q: str):
    """Provides Hypothetical Document Embedding.
    Params:
    - q: the query to provide a Hypothetical document for

    JSON response includes:
    - question: The original question submitted
    - hyde: The expanded hypothetical document (may contain hallucinations)
    """
    hyde_prompt = PromptTemplate(
        template=hyde_prompt_text, input_variables=["question"]
    )

    hyde_chain = hyde_prompt | getLLM()

    return api_response({"question": q, "hyde": hyde_chain.invoke({"question": q})})


@app.get("/context_retrieval")
@guardrail_api
async def context_retrieval(q: str, h: bool = False):
    """Retrieves context documents from vector store.
    Params:
    - q: the query to retrieve context for
    - h: true/false rather to apply HYDE enhancement

    JSON response includes:
    - original_question: the initial query provided
    - hyde: true/false if HYDE was applied
    - question: original question or HYDE enhanced depending on if HYDE enabled
    - context: array of context documents retrieved with following attributes
        - metadata.source
        - metadata.similarity_score
        - page_content
    """

    context_retrieval_chain = (
        retriever_with_scores(VECTOR_STORE)
        | guardrail.similarity_guardrail(settings.SIMILARITY_THRESHOLD)
        | dedupulicate_contexts
    )

    # Enhance with HYDE is specified:
    hq = (await hyde(q))["response"]["hyde"] if h else q

    return api_response(
        {
            "original_question": q,
            "hyde": h,
            "question": hq,
            "context": context_retrieval_chain.invoke({"question": hq}),
        }
    )


class RAGPrompt(BaseModel):
    question: str
    context: str
    answer: Optional[str] = None


@app.get("/llm")
async def llm(llm_prompt: RAGPrompt):
    """Retrieves response generated by LLM."""

    q = llm_prompt.question
    c = llm_prompt.context

    # Initial LLM generation prompt:
    qa_prompt = PromptTemplate(
        template=rag_prompt_llama_text, input_variables=["question", "context"]
    )

    chain = qa_prompt | getLLM()
    llm_response = chain.invoke({"question": q, "context": c})
    return api_response({"answer": llm_response})


@app.put("/evalulate")
@guardrail_api
async def evaluation(llm_response: RAGPrompt):
    """Evaluates the LLM response for accuracy."""

    # Structure definition for evaluation result:
    json_parser = JsonOutputParser(pydantic_object=LLMEvalResult)

    # LLM-as-judge evaluation prompt:
    qa_eval_prompt_with_context = PromptTemplate(
        template=qa_eval_prompt_with_context_text,
        input_variables=["question", "answer", "context"],
        partial_variables={
            "format_instructions": json_parser.get_format_instructions()
        },
    )

    evaluation = qa_eval_prompt_with_context | getLLM() | json_parser
    eval_response = evaluation.invoke(
        {
            "question": llm_response.question,
            "context": llm_response.context,
            "answer": llm_response.answer,
        }
    )
    return api_response({"evaluation": eval_response})


@app.get("/answer")
@guardrail_api
async def answer(q: str, h: bool = True, e: bool = True):
    """Provides an LLM response based on query."""
    # https://towardsdatascience.com/building-a-rag-chain-using-langchain-expression-language-lcel-3688260cad05

    # Context Retrieval
    try:
        context_response = (await context_retrieval(q, h))["response"]
    except Exception as err:
        logger.error(f"Error during context retrieval: {err}")
        logger.error(await context_retrieval(q, h))
        return api_response({"error": "Failed to retrieve any context"})

    # LLM Response
    prompt: RAGPrompt = RAGPrompt(question=q, context=format_docs(context_response))
    llm_response = (await llm(prompt))["response"]

    # LLM Evaluation
    prompt.answer = llm_response["answer"]
    if e:
        evaluation_response = (await evaluation(prompt))["response"]
    else:
        evaluation_response = {"evaluation": {"grade": "", "description": ""}}

    # fmt: off
    return api_response(
        {
            "question": q,
            "hyde": h,
            "answer": llm_response["answer"],
            "context": context_response["context"],
            "evaluation": evaluation_response["evaluation"]
        }
    )
    # fmt: on

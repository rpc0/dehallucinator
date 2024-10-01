from fastapi import FastAPI

from starlette.middleware.cors import CORSMiddleware

from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain import hub
from langchain_core.runnables import chain
from langchain_core.documents import Document
from langchain_community.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser


from operator import itemgetter
from typing import List
import logging
from tqdm import tqdm

import deh.settings as settings
import deh.guardrail as guardrail
from deh.utils import format_context_documents as format_docs
from deh.prompts import (
    qa_eval_prompt_with_context_text,
    LLMEvalResult,
    rag_prompt_llama_text,
)

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
logging.basicConfig(level=logging.INFO)  # logging.basicConfig(level=logging.DEBUG)

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
        global vector_store
        print(f"Vector Store Loaded {vector_store}")

    except Exception as exc:
        print(str(exc))


def initialize_vectorstore(doc_loc: str, doc_filter="**/*.*"):
    """Initialize and cache vector store interface."""
    print("Initializing vector store...")
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
            silent_errors=False,
        )
        data = list(
            tqdm(loader.load(), desc="Loading documents")
        )  # data = loader.load()
        doc_count = len(data)
        logger.info(f"Loaded {doc_count} documents")

        # Split into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.TXT_CHUNK_SIZE, chunk_overlap=settings.TXT_CHUNK_OVERLAP
        )

        # Store text splitter class type for logging:
        TXT_SPLITTER = text_splitter.__class__.__name__

        all_splits = text_splitter.split_documents(data)
        logger.info(f"Split into {len(all_splits)} chunks")

        # Initialize the vector store with GPU embedding function
        model_kwargs = {"device": "cuda"}
        encode_kwargs = {"normalize_embeddings": True}
        vector_store = Chroma.from_documents(
            collection_name="Squad2.0",
            documents=tqdm(all_splits, desc="Vectorizing documents"),  # all_splits,
            embedding=HuggingFaceEmbeddings(
                model_name=settings.EMBEDDING_MODEL,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs,
            ),
            persist_directory=doc_loc + "/cache",
        )
        logger.info(f"Vector Store Loaded {vector_store} and {doc_count} documents.")

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


@chain
def retriever_with_scores(query: str) -> List[Document]:
    """Retrieve documents from vectorstore with similarity score."""
    # https://python.langchain.com/docs/how_to/add_scores_retriever/
    docs, scores = zip(*vector_store.similarity_search_with_score(query))
    for doc, score in zip(docs, scores):
        doc.metadata["similarity_score"] = score

    return docs


@app.get("/answer")
async def answer(question: str):
    """Provides an LLM response based on query."""
    # https://towardsdatascience.com/building-a-rag-chain-using-langchain-expression-language-lcel-3688260cad05
    global vector_store
    print(f"{DOCS_LOADED} documents loaded into vector store.")

    # https://python.langchain.com/v0.1/docs/modules/data_connection/retrievers/vectorstore/
    retriever = vector_store.as_retriever()

    llm = Ollama(
        base_url=settings.OLLAMA_HOST,
        model=settings.LLM_MODEL,
        verbose=True,
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
    )

    try:
        response = rag_chain_context_similarity_exception(retriever, question, llm)
    except guardrail.GuardRailException as exc:
        response = {"error", exc}

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

    # fmt: off
    rag_chain = (
        RunnableParallel(
            context=retriever | format_docs, 
            question=RunnablePassthrough()
        )
        | qa_prompt
        | llm
    )
    # fmt: on

    return rag_chain.invoke(question)


def basic_rag_chain_with_context(retriever, question, llm):
    """Simplest RAG Chain (v0.1) implementation which includes context pass-through."""

    qa_prompt = PromptTemplate(
        template=rag_prompt_llama_text, input_variables=["question", "context"]
    )

    # fmt: off
    rag_chain = (
        RunnableParallel(
            context=retriever | format_docs, 
            question=RunnablePassthrough() )
        | RunnableParallel(
            answer=qa_prompt | llm,
            question=itemgetter("question"),
            context=itemgetter("context"),
        )
    )
    # fmt: on

    return rag_chain.invoke(question)


def rag_chain_context_similarity_exception(retriever, question, llm):
    """Throw exception if below answer_similarity threshold (v1)."""

    qa_prompt = PromptTemplate(
        template=rag_prompt_llama_text, input_variables=["question", "context"]
    )

    # fmt: off
    rag_chain = (
        RunnableParallel(
            context=retriever_with_scores | guardrail.similarity_guardrail(settings.SIMILIARITY_THRESHOLD) | format_docs, 
            question=RunnablePassthrough() )
        | RunnableParallel(
            answer=qa_prompt | llm,
            question=itemgetter("question"),
            context=itemgetter("context"),
        )
    )
    # fmt: on

    return rag_chain.invoke(question)


def rag_chain_with_llm_context_self_evaluation(retriever, question, llm):
    """RAG Chain with exception based on context similarity (stretch-vX)."""

    # Structure definition for evaluation result:
    json_parser = JsonOutputParser(pydantic_object=LLMEvalResult)

    # Evaluation prompt definition:
    qa_eval_prompt_with_context = PromptTemplate(
        template=qa_eval_prompt_with_context_text,
        input_variables=["question", "answer", "context"],
        partial_variables={
            "format_instructions": json_parser.get_format_instructions()
        },
    )

    qa_prompt = hub.pull(settings.LLM_PROMPT)

    # fmt: off
    rag_chain = (
        RunnableParallel(
            context = retriever | format_docs, 
            question = RunnablePassthrough() )
        | RunnableParallel(
            answer= qa_prompt | llm, 
            question = itemgetter("question"), 
            context = itemgetter("context") )
        | RunnableParallel(
            answer = itemgetter("answer"),
            question = itemgetter("question"),
            context = itemgetter("context"),
            evaluation = qa_eval_prompt_with_context | llm | json_parser
        )
    )
    # fmt: on

    return rag_chain.invoke(question)

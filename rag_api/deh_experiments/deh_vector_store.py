"""

Contains data and methods related to the vector store (Chroma is used here).

"""

# ==========================================================================
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.documents import Document
from deh_semantic_chunking import SemanticChunker
from langchain_ollama import OllamaEmbeddings
import deh_globals

CHROMA_ROOT = "../../../deh_data_results/chroma"     # Set to your own root chroma folder
ollama_embedding_model = "avr/sfr-embedding-mistral"
embeddings = OllamaEmbeddings(model=ollama_embedding_model)
persist_directory = f"{CHROMA_ROOT}/chroma/chroma_deh_rag_db_k{deh_globals.VECTOR_STORE_TOP_K}_cs{deh_globals.CHUNK_SIZE}"


# ==========================================================================
# Gets the vector store for an experiment

def get_vector_store(prefix, chunking_method):

    # TODO: currently, vector store and collection name are the same, and
    #       each collection is stored in its own chromadb.
    # Ultimately, all collections can be stored in the same chromadb.
    collection_name = f"{prefix}_{chunking_method}"
    print(f"Will now get the following vector store: {persist_directory}" + f"_{chunking_method}")
    return Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_directory + f"_{chunking_method}"
    )


# ==========================================================================
# Gets a splitter, depending on the chunking method.

def get_splitter(chunking_method, chunk_size, chunk_overlap):

    if chunking_method == "naive":
        return CharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    if chunking_method in ["per_context", "per_article"]:
        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    elif chunking_method == "semantic":
        return SemanticChunker(
            # embeddings=embeddings
            # breakpoint_threshold_type="standard_deviation"
            # chunk_size=chunk_size,
            # chunk_overlap=chunk_overlap
        )
    else:
        return None


# ==========================================================================
# Chunks all the contexts contained in the first parameter, which is a list.
# Note that the term "context" designates the SQuAD contexts, as read from
# the raw sqaud file (not to be confused with the contexts returned by the
# vector store).

def chunk_contexts(contexts, chunking_method, chunk_size, chunk_overlap, dataset=None):

    # The splitter is not needed for "pseudo_semantic" chunking.
    # get_splitter returns None in that case.
    splitter = get_splitter(chunking_method, chunk_size, chunk_overlap)

    if chunking_method == "naive":
        all_contexts = "\n\n".join(contexts)
        chunks = splitter.create_documents([all_contexts])  
    elif chunking_method == "per_context":
        chunks = []
        for context in contexts:
            context_specific_chunks = splitter.create_documents([context])
            for chunk in context_specific_chunks:
                chunks.append(chunk)
    elif chunking_method == "per_article":
        chunks = []
        for article in dataset:
            article_contexts = []

            for p in article['paragraphs']:
                article_contexts.append(p["context"])

            all_article_contexts = "\n\n".join(article_contexts)
            article_chunks = splitter.create_documents([all_article_contexts]) 
            for article_chunk in article_chunks:
                chunks.append(article_chunk)
    elif chunking_method == "semantic":
        all_contexts = "\n\n".join(contexts)
        chunks = splitter.split_text(all_contexts)
        chunks = [Document(page_content=chunk, metadata={"source": "squad"}) for chunk in chunks]
    elif chunking_method == "pseudo_semantic":
        chunks = [Document(page_content=context, metadata={"source": "squad"}) for context in contexts]

    return chunks


# ==========================================================================
# Add all the chunks that result from chunking to the vector store.

def add_chunks_to_vector_store(chunks, vector_store):
    ids = [str(i) for i in list(range(len(chunks)))]
    for i, chunk in enumerate(chunks):
        if i % 10 == 0:
            print(f"Adding chunk {i} to the vector store...")
        vector_store.add_documents(documents=[chunk], ids=[ids[i]])


# ==========================================================================
# Chunks all the chunks contained in the parameter "squad_contexts". Chunking
# is carried out for all chunking methods.

def chunk_squad_dataset(squad_contexts, dataset, chunk_size=400, chunk_overlap=50):

    print(f"Creating contexts for the dataset...")
    chunking_methods = ["naive", "per_context", "per_article", "semantic", "pseudo_semantic"]

    for chunking_method in chunking_methods:

        print(f"Chunking method: {chunking_method}")

        vector_store = get_vector_store("deh_rag", chunking_method)
        chunks = chunk_contexts(squad_contexts, chunking_method, chunk_size, chunk_overlap, dataset)

        for chunk in chunks:
            chunk.metadata = {"source": "squad"}
        print(f"Number of chunks --> {len(chunks)}\n")

        add_chunks_to_vector_store(chunks, vector_store)

# ==========================================================================
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_experimental.text_splitter import SemanticChunker
from langchain_core.documents import Document
from deh_semantic_chunking import SemanticChunker
from langchain_ollama import OllamaEmbeddings

# DATA_ROOT = "../../../deh_data_results/data"         # Set to your own data folder
CHROMA_ROOT = "../../../deh_data_results/chroma"     # Set to your own chroma folder
ollama_embedding_model = "avr/sfr-embedding-mistral"
embeddings = OllamaEmbeddings(model=ollama_embedding_model)
persist_directory = f"{CHROMA_ROOT}/chroma/chroma_deh_rag_db"


# ==========================================================================
def get_vector_store(prefix, chunking_method):

    collection_name = f"{prefix}_{chunking_method}"
    return Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_directory + f"_{chunking_method}"
    )


# ==========================================================================
def get_splitter(chunking_method, chunk_size, chunk_overlap):

    if chunking_method in ["naive", "per_context", "per_article"]:
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
def add_chunks_to_vector_store(chunks, vector_store):
    ids = [str(i) for i in list(range(len(chunks)))]
    for i, chunk in enumerate(chunks):
        if i % 10 == 0:
            print(f"Adding chunk {i} to the vector store...")
        vector_store.add_documents(documents=[chunk], ids=[ids[i]])


# ==========================================================================
def chunk_squad_dataset(squad_contexts, dataset, chunk_size=400, chunk_overlap=50):

    print(f"Creating contexts for the dataset...")
    chunking_methods = ["naive", "per_context", "per_article", "semantic", "pseudo_semantic"]

    for chunking_method in chunking_methods:

        # # # TODO: Remove this IF STATEMENT, once semantic chunking works !!!!
        if not chunking_method == "naive":
            continue
        
        print(f"Chunking method: {chunking_method}")
        # collection_name = f"deh_rag_{chunking_method}"
        # print(f"Collection name: {collection_name}")

        vector_store = get_vector_store("deh_rag", chunking_method)
        chunks = chunk_contexts(squad_contexts, chunking_method, chunk_size, chunk_overlap, dataset)

        for chunk in chunks:
            chunk.metadata = {"source": "squad"}
        print(f"Number of chunks --> {len(chunks)}\n")

        add_chunks_to_vector_store(chunks, vector_store)

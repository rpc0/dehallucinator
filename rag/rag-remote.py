# Load web page
import argparse

from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Embed and store
from langchain.vectorstores import Chroma
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.embeddings import GPT4AllEmbeddings  # Or GPT4All
from langchain.embeddings import OllamaEmbeddings  # We can also try Ollama embeddings

from langchain_community.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler


def main():

    # Load context files from directory:
    loader = DirectoryLoader(path="data", glob="**/*.context")
    data = loader.load()
    print(f"Loaded {len(data)} documents")

    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=100)
    all_splits = text_splitter.split_documents(data)
    print(f"Split into {len(all_splits)} chunks")

    vectorstore = Chroma.from_documents(
        documents=all_splits, embedding=HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    )

    # RAG prompt
    from langchain import hub

    QA_CHAIN_PROMPT = hub.pull("rlm/rag-prompt-llama")

    # LLM
    llm = Ollama(
        base_url="http://localhost:7869",
        model="llava-phi3:latest",
        verbose=True,
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
    )
    print(f"Loaded LLM model {llm.model}")

    # QA chain
    from langchain.chains import RetrievalQA

    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=vectorstore.as_retriever(),
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
    )

    # Ask a question
    question = f"What is hallucination in LLMs?"
    result = qa_chain({"query": question})

    # print(result)


if __name__ == "__main__":
    main()

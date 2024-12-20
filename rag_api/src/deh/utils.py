from langchain_core.runnables import chain
from langchain_core.documents import Document

from typing import List

import deh.settings as settings


def format_context_documents(inputs):
    """Formats array of contexts into string.
    Input:
    - documents in key 'context'
    """
    return "\n\n ------------".join(doc.page_content for doc in inputs["context"])


def retriever_with_scores(vector_store, k=settings.CONTEXT_DOCUMENTS_RETRIEVED):
    """Retrieve documents from vectorstore with similarity score."""

    @chain
    def retrieved_docs(inputs: str) -> List[Document]:
        # https://python.langchain.com/docs/how_to/add_scores_retriever/

        search_results = vector_store.similarity_search_with_score(
            inputs["question"], k=k
        )

        # GUARD-STATEMENT: No results
        if len(search_results) == 0:
            return []

        docs, scores = zip(*search_results)
        for doc, score in zip(docs, scores):
            doc.metadata["similarity_score"] = score

        return docs

    return retrieved_docs


def dedupulicate_contexts(docs):
    # TODO: Given potential for overlapping contexts this could be needed in future.
    return docs

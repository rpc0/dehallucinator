from langchain_core.runnables import chain
from langchain_core.documents import Document
from typing import List

from deh import settings


class GuardRailException(Exception):
    pass


def similarity_guardrail(threshold):
    """Guard-rail: Raise exception if no contexts found above threshold."""

    @chain
    def exception_if_no_results(docs) -> List[Document]:
        filtered_docs = [
            doc for doc in docs if doc.metadata["similarity_score"] < threshold
        ]
        if len(filtered_docs) <= 0:
            raise GuardRailException(
                f"No contexts found at similarity threshold: {settings.SIMILARITY_THRESHOLD}"
            )

        return filtered_docs

    return exception_if_no_results

def format_context_documents(docs):
    return "\n\n ------------".join(doc.page_content for doc in docs)

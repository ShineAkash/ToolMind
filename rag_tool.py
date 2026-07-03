"""
RAG Tool — upload documents and query them via semantic search.
Uses FAISS (local, no API key) + HuggingFace sentence-transformers embeddings.
"""

import os
from langchain_core.tools import tool
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader

# ── Shared in-memory vector store (rebuilt when docs are uploaded) ────────────
_vectorstore = None
_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


def load_documents(file_paths: list[str]) -> int:
    """
    Load a list of PDF or TXT file paths into the in-memory FAISS vector store.
    Returns the number of chunks indexed.
    """
    global _vectorstore

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    all_docs = []

    for path in file_paths:
        ext = os.path.splitext(path)[-1].lower()
        if ext == ".pdf":
            loader = PyPDFLoader(path)
        elif ext == ".txt":
            loader = TextLoader(path, encoding="utf-8")
        else:
            continue
        docs = loader.load()
        all_docs.extend(splitter.split_documents(docs))

    if not all_docs:
        return 0

    if _vectorstore is None:
        _vectorstore = FAISS.from_documents(all_docs, _embeddings)
    else:
        _vectorstore.add_documents(all_docs)

    return len(all_docs)


def clear_documents():
    """Clear the vector store."""
    global _vectorstore
    _vectorstore = None


def has_documents() -> bool:
    return _vectorstore is not None


@tool
def rag_search(query: str) -> str:
    """
    Search the uploaded documents for information relevant to the query.
    Use this tool whenever the user asks a question that might be answered
    by the documents they have uploaded.
    """
    if _vectorstore is None:
        return "No documents have been uploaded yet. Please upload a PDF or TXT file first."

    results = _vectorstore.similarity_search(query, k=4)
    if not results:
        return "No relevant information found in the uploaded documents."

    context = "\n\n---\n\n".join(
        f"[Source: {doc.metadata.get('source', 'unknown')}, page {doc.metadata.get('page', '?')}]\n{doc.page_content}"
        for doc in results
    )
    return f"Relevant excerpts from uploaded documents:\n\n{context}"

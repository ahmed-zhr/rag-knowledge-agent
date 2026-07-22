"""Vector store layer - one Chroma collection per business domain.

Keeping domains in separate collections is a deliberate governance choice:
an agent answering maintenance questions can never surface HR content, even
by accident. Isolation happens at storage level, not at prompt level.
"""
from __future__ import annotations

from langchain_chroma import Chroma
from langchain_core.documents import Document

import config
from src.embeddings import get_embeddings


def get_store(domain: str, embeddings=None) -> Chroma:
    """Return the persistent Chroma collection for a domain."""
    if domain not in config.DOMAINS:
        raise ValueError(f"Unknown domain '{domain}'. Expected one of {config.DOMAINS}.")
    return Chroma(
        collection_name=f"kb_{domain}",
        embedding_function=embeddings or get_embeddings(),
        persist_directory=str(config.VECTOR_DB_DIR),
    )


def add_documents(domain: str, documents: list[Document], embeddings=None) -> int:
    """Add pre-chunked documents to a domain's knowledge base."""
    if not documents:
        return 0
    get_store(domain, embeddings).add_documents(documents)
    return len(documents)


def get_retriever(domain: str, embeddings=None):
    """Similarity retriever scoped to a single domain."""
    return get_store(domain, embeddings).as_retriever(search_kwargs={"k": config.TOP_K})


def collection_stats(embeddings=None) -> dict[str, int]:
    """Number of stored chunks per domain (for the UI sidebar)."""
    stats = {}
    for domain in config.DOMAINS:
        try:
            stats[domain] = get_store(domain, embeddings)._collection.count()
        except Exception:
            stats[domain] = 0
    return stats

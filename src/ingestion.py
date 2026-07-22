"""Document ingestion - load, chunk and index files into the knowledge base.

Expected layout (one sub-folder per domain):
    data/sample_docs/operations/*.md
    data/sample_docs/maintenance/*.md
    data/sample_docs/hr/*.md

Every chunk carries metadata (domain, source file, ingestion date) so answers
can cite their sources and every piece of knowledge stays traceable.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config
from src import vectorstore

SUPPORTED_SUFFIXES = {".md", ".txt"}


def load_domain_files(domain: str, base_dir: Path | None = None) -> list[Document]:
    """Read every supported file of a domain folder into raw Documents."""
    folder = (base_dir or config.DATA_DIR) / domain
    docs: list[Document] = []
    if not folder.is_dir():
        return docs
    for path in sorted(folder.iterdir()):
        if path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        docs.append(Document(
            page_content=path.read_text(encoding="utf-8"),
            metadata={
                "domain": domain,
                "source": path.name,
                "origin": "document",
                "ingested_on": date.today().isoformat(),
            },
        ))
    return docs


def chunk_documents(docs: list[Document]) -> list[Document]:
    """Split documents into overlapping chunks sized for retrieval."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(docs)


def ingest_domain(domain: str, base_dir: Path | None = None, embeddings=None) -> dict:
    """Full pipeline for one domain: load -> chunk -> index. Returns a report."""
    raw = load_domain_files(domain, base_dir)
    chunks = chunk_documents(raw)
    added = vectorstore.add_documents(domain, chunks, embeddings)
    return {"domain": domain, "files": len(raw), "chunks_indexed": added}


def ingest_all(base_dir: Path | None = None, embeddings=None) -> list[dict]:
    """Ingest every domain folder. Used by scripts/ingest.py."""
    return [ingest_domain(d, base_dir, embeddings) for d in config.DOMAINS]

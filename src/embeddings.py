"""Embedding provider factory (huggingface | ollama | openai)."""
from __future__ import annotations

from functools import lru_cache

import config


@lru_cache(maxsize=1)
def get_embeddings():
    provider = config.EMBEDDING_PROVIDER.lower()

    if provider == "huggingface":
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)

    if provider == "ollama":
        from langchain_ollama import OllamaEmbeddings
        return OllamaEmbeddings(model=config.OLLAMA_EMBEDDING_MODEL)

    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model=config.OPENAI_EMBEDDING_MODEL)

    raise ValueError(
        f"Unknown EMBEDDING_PROVIDER '{config.EMBEDDING_PROVIDER}'. "
        "Use 'huggingface', 'ollama' or 'openai'."
    )
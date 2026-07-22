"""LLM provider factory.

The rest of the codebase never imports a vendor SDK directly: it calls
`get_llm()` and receives a LangChain chat model. Swapping providers is a
one-line change in `.env` (LLM_PROVIDER=openai|ollama).
"""
from __future__ import annotations

import config


def get_llm():
    """Return the configured chat model."""
    provider = config.LLM_PROVIDER.lower()

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=config.OPENAI_MODEL, temperature=config.LLM_TEMPERATURE)

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(model=config.OLLAMA_MODEL, temperature=config.LLM_TEMPERATURE)

    raise ValueError(
        f"Unknown LLM_PROVIDER '{config.LLM_PROVIDER}'. Use 'openai' or 'ollama'."
    )

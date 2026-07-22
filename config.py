"""Central configuration for the RAG Knowledge Agent.

All tunable settings live here. Values can be overridden through a `.env`
file (see `.env.example`) so the code never needs editing between machines.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --------------------------------------------------------------- paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data" / "sample_docs"
VECTOR_DB_DIR = PROJECT_ROOT / os.getenv("VECTOR_DB_DIR", "chroma_db")
AUDIT_LOG_PATH = PROJECT_ROOT / "knowledge_audit.jsonl"

# --------------------------------------------------------------- domains
# One knowledge space per business domain. Retrieval is always scoped to a
# single domain so agents never mix knowledge across departments.
DOMAINS = ["operations", "maintenance", "hr"]

# --------------------------------------------------------------- chunking
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))


EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "ollama")

OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
# --------------------------------------------------------------- embeddings
# Runs locally on CPU, free, no API key required.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# --------------------------------------------------------------- retrieval
TOP_K = int(os.getenv("TOP_K", "4"))

# --------------------------------------------------------------- LLM provider
# "openai" (needs OPENAI_API_KEY) or "ollama" (needs a local Ollama install).
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))

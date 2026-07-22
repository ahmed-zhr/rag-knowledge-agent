"""Human-in-the-loop memory - the governance core of this project.

The agent can learn from conversations, but NOTHING enters the knowledge base
without explicit human approval. The loop:

    1. PROPOSE - the LLM distills the conversation into a factual note
    2. REVIEW  - the user reads and freely edits the proposed note
    3. APPROVE - only then is the note embedded and stored
    4. AUDIT   - every approval is appended to a JSONL audit trail

Why it matters: unsupervised "self-learning" is how a knowledge base gets
polluted with errors. A human gate keeps the base trustworthy, and the audit
trail makes every stored fact traceable to a decision.
"""
from __future__ import annotations

import json
from datetime import datetime

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

import config
from src import vectorstore
from src.llm import get_llm

SUMMARY_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You distill conversations into knowledge-base notes for the {domain} "
     "department.\n"
     "Write a short factual note (5-10 lines) capturing ONLY reusable "
     "knowledge from this conversation: decisions, procedures, facts, "
     "corrections. Exclude greetings, questions and anything uncertain. "
     "Write in the third person, no preamble."),
    ("human", "Conversation transcript:\n\n{transcript}"),
])


def render_transcript(history: list[dict]) -> str:
    """Flatten UI history into a plain transcript for the summarizer."""
    return "\n".join(
        f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
        for m in history
    )


def propose_summary(history: list[dict], domain: str, llm=None) -> str:
    """Step 1 - PROPOSE: ask the LLM for a distilled knowledge note."""
    chain = SUMMARY_PROMPT | (llm or get_llm()) | StrOutputParser()
    return chain.invoke({"domain": domain,
                         "transcript": render_transcript(history)}).strip()


def approve_and_store(note: str, domain: str, approved_by: str = "user",
                      embeddings=None) -> dict:
    """Steps 3+4 - APPROVE & AUDIT: store the (possibly edited) note.

    The note is indexed like any document, but its metadata marks it as
    conversation-born and approved, so answers can disclose their origin."""
    note = note.strip()
    if not note:
        raise ValueError("Cannot store an empty note.")

    timestamp = datetime.now().isoformat(timespec="seconds")
    source_id = f"conversation_note_{timestamp.replace(':', '-')}"

    vectorstore.add_documents(domain, [Document(
        page_content=note,
        metadata={
            "domain": domain,
            "source": source_id,
            "origin": "conversation",
            "approved_by": approved_by,
            "approved_at": timestamp,
        },
    )], embeddings)

    with open(config.AUDIT_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "approved_at": timestamp,
            "approved_by": approved_by,
            "domain": domain,
            "source_id": source_id,
            "note": note,
        }, ensure_ascii=False) + "\n")

    return {"source_id": source_id, "approved_at": timestamp}


def read_audit_log(limit: int = 20) -> list[dict]:
    """Latest audit entries (for the UI transparency panel)."""
    if not config.AUDIT_LOG_PATH.is_file():
        return []
    lines = config.AUDIT_LOG_PATH.read_text(encoding="utf-8").strip().splitlines()
    return [json.loads(line) for line in lines[-limit:]][::-1]

"""The RAG chain - retrieve, ground, answer, cite.

Flow for every question:
    1. retrieve the most relevant chunks from the domain's knowledge base
    2. inject them (with source names) into a grounded prompt
    3. pass the recent conversation history for continuity
    4. answer ONLY from the provided context, citing sources

The model is explicitly told to say when the knowledge base does not contain
the answer - a grounded "I don't know" beats a confident guess.
"""
from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from src import vectorstore
from src.llm import get_llm

SYSTEM_PROMPT = """You are the internal knowledge assistant for the {domain} \
department of MetroLink Transit.

Answer the user's question using ONLY the context below. Rules:
- Ground every statement in the context; never invent facts.
- Cite the source file after each fact, like [source: file.md].
- If the context does not contain the answer, say so clearly and suggest what \
information should be added to the knowledge base.
- Be concise and professional.

Context:
{context}"""


def format_context(docs: list[Document]) -> str:
    """Render retrieved chunks with their source names for citation."""
    if not docs:
        return "(no relevant documents found)"
    blocks = []
    for doc in docs:
        src = doc.metadata.get("source", "unknown")
        origin = doc.metadata.get("origin", "document")
        label = src if origin == "document" else f"{src} (approved conversation note)"
        blocks.append(f"--- {label} ---\n{doc.page_content}")
    return "\n\n".join(blocks)


def to_lc_history(history: list[dict], max_turns: int = 6):
    """Convert UI history [{'role','content'}] to LangChain messages.

    Only the last `max_turns` messages are sent: the point is conversational
    continuity, not replaying the whole session (which would waste tokens and
    drown the retrieved context)."""
    msgs = []
    for m in history[-max_turns:]:
        cls = HumanMessage if m["role"] == "user" else AIMessage
        msgs.append(cls(content=m["content"]))
    return msgs


def answer(question: str, domain: str, history: list[dict] | None = None,
           llm=None, embeddings=None) -> dict:
    """Run the full RAG loop. Returns the answer plus the sources used."""
    docs = vectorstore.get_retriever(domain, embeddings).invoke(question)

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder("history"),
        ("human", "{question}"),
    ])
    chain = prompt | (llm or get_llm()) | StrOutputParser()

    text = chain.invoke({
        "domain": domain,
        "context": format_context(docs),
        "history": to_lc_history(history or []),
        "question": question,
    })
    sources = sorted({d.metadata.get("source", "unknown") for d in docs})
    return {"answer": text, "sources": sources, "retrieved": len(docs)}

# RAG Knowledge Agent

**An internal documentation assistant with a human-governed knowledge base.**

Most RAG demos stop at "chat with your PDF". This one addresses the problem that
shows up once a knowledge assistant is actually deployed inside a company:
**who decides what the agent is allowed to know?**

The agent answers questions from a company's internal documentation, cites its
sources, and can *learn* from conversations — but nothing ever enters its
knowledge base without an explicit human approval, and every approval is logged.

---

## Why this design

Three decisions drive the architecture:

**1. Human-in-the-loop knowledge growth.** After a conversation, the agent can
distill what was discussed into a knowledge note. The note is *proposed*, the
user *edits* it freely, and only an explicit approval stores it. Unsupervised
self-learning is how a knowledge base silently fills with errors; a human gate
keeps it trustworthy.

**2. Domain isolation at storage level.** Each department (operations,
maintenance, HR) lives in its own vector collection. An agent answering a
maintenance question cannot surface HR content — the separation is structural,
not a prompt instruction that a model might ignore.

**3. Traceability by default.** Every answer cites its source files. Every
approved note is appended to a JSONL audit trail with a timestamp — so any
stored fact can be traced back to the decision that put it there.

---

## Architecture

```
                     ┌──────────────────────────────┐
  documents ────────▶│  INGESTION                   │
  (per domain)       │  load → chunk → embed        │
                     └───────────────┬──────────────┘
                                     ▼
                     ┌──────────────────────────────┐
                     │  VECTOR STORE (Chroma)       │◀── approved note
                     │  one collection per domain   │           ▲
                     └───────────────┬──────────────┘           │
                                     ▼                          │
  question ─────────▶┌──────────────────────────────┐   ┌───────┴────────┐
                     │  RAG CHAIN                   │   │ HUMAN-IN-THE-  │
                     │  retrieve → ground → answer  │──▶│ LOOP           │
                     │  + source citations          │   │ propose → edit │
                     └──────────────────────────────┘   │ → approve      │
                                                        └────────────────┘
```

**Stack:** Python · LangChain · Chroma · HuggingFace embeddings (local, CPU) ·
OpenAI or Ollama for generation · Streamlit UI

---

## Quickstart

```bash
git clone https://github.com/<your-username>/rag-knowledge-agent.git
cd rag-knowledge-agent

python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env        # then add your OPENAI_API_KEY
                            # (or set LLM_PROVIDER=ollama to run fully local)

python scripts/ingest.py    # index the sample documents
streamlit run app.py
```

Embeddings run locally on CPU — no API key needed for the retrieval half.
Set `LLM_PROVIDER=ollama` to run the whole system offline and free.

---

## Using it

**Ask a question.** Pick a department in the sidebar and ask. The agent
retrieves the relevant passages, answers from them only, and cites the files it
used. If the knowledge base doesn't cover the question, it says so and suggests
what should be added — rather than guessing.

**Teach it something.** After a conversation, click *Propose a note*. The agent
drafts a factual summary of what was established. Edit it, then approve — the
note joins the knowledge base and becomes retrievable in future conversations,
tagged as conversation-born and logged in the audit trail (visible in the
sidebar).

---

## Project structure

```
config.py              # all settings, overridable via .env
src/
  vectorstore.py       # Chroma collections, one per domain
  ingestion.py         # load → chunk → index pipeline
  rag_chain.py         # retrieval + grounded prompt + citations
  memory_loop.py       # propose / review / approve / audit
  llm.py               # provider factory (OpenAI | Ollama)
app.py                 # Streamlit interface
scripts/ingest.py      # CLI ingestion
data/sample_docs/      # synthetic demo documents
```

---

## About the data

The system is designed for real internal enterprise documentation. The
documents in this repository are **entirely synthetic** — they describe a
fictional public transport operator ("MetroLink Transit") and contain no data
from any real organization. The `.gitignore` excludes `data/private/` and
`data/internal/` so that real documents stay out of version control by default.

---

## Possible extensions

- Pluggable document sources (SharePoint / Microsoft Lists connectors)
- Role-based access control on top of domain isolation
- Retrieval evaluation harness (recall@k on a labelled question set)
- Reranking to sharpen retrieval on larger corpora

---

## License

MIT

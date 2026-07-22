"""RAG Knowledge Agent — Streamlit UI.

Three panels in one page:
    - sidebar : domain selector, knowledge-base stats, audit trail
    - main    : chat grounded on the selected domain's knowledge base
    - learning: propose / review / approve loop to grow the knowledge base

Run with:  streamlit run app.py
"""
import streamlit as st

import config
from src import ingestion, memory_loop, rag_chain, vectorstore

st.set_page_config(page_title="RAG Knowledge Agent", page_icon="📚", layout="wide")

st.markdown("""
<style>
  .main-title { font-weight:700; font-size:1.7rem; margin-bottom:0; }
  .subtitle { color:#5B6B7B; font-size:0.95rem; margin-top:0.15rem; }
  .stButton>button[kind="primary"] { border:none; }
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------- state
def _init(key, value):
    if key not in st.session_state:
        st.session_state[key] = value


_init("history", [])          # [{'role': 'user'|'assistant', 'content': str}]
_init("proposed_note", None)  # note awaiting human review
_init("last_saved", None)


# ----------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("### Knowledge agent")
    domain = st.selectbox("Department", config.DOMAINS, format_func=str.title)

    stats = vectorstore.collection_stats()
    st.caption("Chunks in knowledge base: "
               + " · ".join(f"{d} {n}" for d, n in stats.items()))

    if sum(stats.values()) == 0:
        st.warning("Knowledge base is empty.")
        if st.button("Ingest sample documents", use_container_width=True):
            with st.spinner("Indexing sample documents..."):
                report = ingestion.ingest_all()
            st.success(" · ".join(f"{r['domain']}: {r['chunks_indexed']}"
                                  for r in report))
            st.rerun()

    if st.button("New conversation", use_container_width=True):
        st.session_state.history = []
        st.session_state.proposed_note = None
        st.rerun()

    with st.expander("Audit trail (approved notes)"):
        entries = memory_loop.read_audit_log()
        if not entries:
            st.caption("No approved notes yet.")
        for e in entries:
            st.markdown(f"**{e['approved_at']}** · {e['domain']}")
            st.caption(e["note"][:180] + ("..." if len(e["note"]) > 180 else ""))


# ----------------------------------------------------------------- header
st.markdown('<p class="main-title">RAG Knowledge Agent</p>', unsafe_allow_html=True)
st.markdown(f'<p class="subtitle">Grounded answers from the '
            f'<b>{domain.title()}</b> knowledge base — with human-approved '
            f'memory. Sources are cited; nothing is stored without review.</p>',
            unsafe_allow_html=True)

# ----------------------------------------------------------------- chat
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if question := st.chat_input("Ask a question about " + domain):
    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Searching the knowledge base..."):
            try:
                result = rag_chain.answer(question, domain,
                                          st.session_state.history[:-1])
                reply = result["answer"]
                if result["sources"]:
                    reply += "\n\n*Sources: " + ", ".join(result["sources"]) + "*"
            except Exception as exc:  # surface config errors kindly
                reply = f"⚠️ {exc}"
        st.markdown(reply)
    st.session_state.history.append({"role": "assistant", "content": reply})

# ----------------------------------------------------------------- learning loop
st.markdown("---")
st.markdown("#### Teach the agent (human-in-the-loop)")
st.caption("The agent can distill this conversation into a knowledge note. "
           "You review and edit it — nothing is stored without your approval.")

col1, col2 = st.columns([1, 3])
with col1:
    disabled = len(st.session_state.history) < 2
    if st.button("Propose a note", type="primary",
                 disabled=disabled, use_container_width=True):
        with st.spinner("Summarizing the conversation..."):
            try:
                st.session_state.proposed_note = memory_loop.propose_summary(
                    st.session_state.history, domain)
            except Exception as exc:
                st.error(str(exc))

with col2:
    if st.session_state.proposed_note is not None:
        edited = st.text_area("Review and edit before approval:",
                              value=st.session_state.proposed_note, height=160)
        a, b = st.columns(2)
        if a.button("Approve & add to knowledge base",
                    type="primary", use_container_width=True):
            receipt = memory_loop.approve_and_store(edited, domain)
            st.session_state.proposed_note = None
            st.session_state.last_saved = receipt
            st.rerun()
        if b.button("Discard", use_container_width=True):
            st.session_state.proposed_note = None
            st.rerun()

if st.session_state.last_saved:
    st.success(f"Note stored in **{domain}** knowledge base "
               f"({st.session_state.last_saved['source_id']}). "
               "It is now retrievable and logged in the audit trail.")
    st.session_state.last_saved = None

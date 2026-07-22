"""RAG Knowledge Agent - interface Streamlit (en francais).

Trois zones dans une page :
    - barre laterale : choix du departement, statistiques, journal d'audit
    - principale     : chat ancre sur la base du departement selectionne
    - apprentissage  : boucle proposer / valider / approuver

Lancement :  streamlit run app.py
"""
import streamlit as st

import config
from src import ingestion, memory_loop, rag_chain, vectorstore

st.set_page_config(page_title="Agent de connaissance RAG", page_icon="📚", layout="wide")

st.markdown("""
<style>
  .main-title { font-weight:700; font-size:1.7rem; margin-bottom:0; }
  .subtitle { color:#5B6B7B; font-size:0.95rem; margin-top:0.15rem; }
  .stButton>button[kind="primary"] { border:none; }
</style>
""", unsafe_allow_html=True)

DOMAIN_LABELS = {"operations": "Exploitation", "maintenance": "Maintenance", "hr": "RH"}


def domain_label(d: str) -> str:
    return DOMAIN_LABELS.get(d, d.title())


# ----------------------------------------------------------------- etat
def _init(key, value):
    if key not in st.session_state:
        st.session_state[key] = value


_init("history", [])
_init("proposed_note", None)
_init("last_saved", None)


# ----------------------------------------------------------------- barre laterale
with st.sidebar:
    st.markdown("### Agent de connaissance")
    domain = st.selectbox("Departement", config.DOMAINS, format_func=domain_label)

    stats = vectorstore.collection_stats()
    st.caption("Fragments dans la base : "
               + " · ".join(f"{domain_label(d)} {n}" for d, n in stats.items()))

    if sum(stats.values()) == 0:
        st.warning("La base de connaissance est vide.")
        if st.button("Indexer les documents d'exemple", use_container_width=True):
            with st.spinner("Indexation des documents..."):
                report = ingestion.ingest_all()
            st.success(" · ".join(f"{domain_label(r['domain'])} : {r['chunks_indexed']}"
                                  for r in report))
            st.rerun()

    if st.button("Nouvelle conversation", use_container_width=True):
        st.session_state.history = []
        st.session_state.proposed_note = None
        st.rerun()

    with st.expander("Journal d'audit (notes validees)"):
        entries = memory_loop.read_audit_log()
        if not entries:
            st.caption("Aucune note validee pour l'instant.")
        for e in entries:
            st.markdown(f"**{e['approved_at']}** · {domain_label(e['domain'])}")
            st.caption(e["note"][:180] + ("..." if len(e["note"]) > 180 else ""))


# ----------------------------------------------------------------- en-tete
st.markdown('<p class="main-title">Agent de connaissance RAG</p>', unsafe_allow_html=True)
st.markdown(f'<p class="subtitle">Reponses ancrees sur la base de connaissance '
            f'<b>{domain_label(domain)}</b>, avec memoire validee par un humain. '
            f'Les sources sont citees ; rien n\'est enregistre sans relecture.</p>',
            unsafe_allow_html=True)

# ----------------------------------------------------------------- chat
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if question := st.chat_input("Posez une question sur : " + domain_label(domain)):
    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Recherche dans la base de connaissance..."):
            try:
                result = rag_chain.answer(question, domain,
                                          st.session_state.history[:-1])
                reply = result["answer"]
                if result["sources"]:
                    reply += "\n\n*Sources : " + ", ".join(result["sources"]) + "*"
            except Exception as exc:
                reply = f"⚠️ {exc}"
        st.markdown(reply)
    st.session_state.history.append({"role": "assistant", "content": reply})

# ----------------------------------------------------------------- boucle d'apprentissage
st.markdown("---")
st.markdown("#### Enseigner a l'agent (validation humaine)")
st.caption("L'agent peut resumer cette conversation en une note de connaissance. "
           "Vous la relisez et la modifiez ; rien n'est enregistre sans votre validation.")

col1, col2 = st.columns([1, 3])
with col1:
    disabled = len(st.session_state.history) < 2
    if st.button("Proposer une note", type="primary",
                 disabled=disabled, use_container_width=True):
        with st.spinner("Synthese de la conversation..."):
            try:
                st.session_state.proposed_note = memory_loop.propose_summary(
                    st.session_state.history, domain)
            except Exception as exc:
                st.error(str(exc))

with col2:
    if st.session_state.proposed_note is not None:
        edited = st.text_area("Relisez et modifiez avant validation :",
                              value=st.session_state.proposed_note, height=160)
        a, b = st.columns(2)
        if a.button("Valider et ajouter a la base",
                    type="primary", use_container_width=True):
            receipt = memory_loop.approve_and_store(edited, domain)
            st.session_state.proposed_note = None
            st.session_state.last_saved = receipt
            st.rerun()
        if b.button("Abandonner", use_container_width=True):
            st.session_state.proposed_note = None
            st.rerun()

if st.session_state.last_saved:
    st.success(f"Note enregistree dans la base **{domain_label(domain)}** "
               f"({st.session_state.last_saved['source_id']}). "
               "Elle est desormais consultable et tracee dans le journal d'audit.")
    st.session_state.last_saved = None
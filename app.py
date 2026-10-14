from __future__ import annotations

import hashlib
import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from care_docs.rag import (
    SAFETY_NOTICE,
    TfidfRetriever,
    chunk_documents,
    extract_document,
    generate_answer,
)

load_dotenv(Path(__file__).with_name(".env"))
st.set_page_config(page_title="CareDocs | Medical document Q&A", page_icon="✚", layout="wide", initial_sidebar_state="expanded")


def setting(name: str, default: str = "") -> str:
    """Read optional settings from environment or Streamlit Cloud secrets."""
    value = os.getenv(name)
    if value:
        return value
    try:
        return str(st.secrets.get(name, default))
    except Exception:
        return default

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
:root{--ink:#1b2d2a;--muted:#72817c;--teal:#227d74;--line:#e4ece8}
html,body,[class*="css"]{font-family:'DM Sans',sans-serif}
.stApp{background:linear-gradient(145deg,#f4faf8 0%,#f7f9f7 55%,#f5f7f4 100%)}
.block-container{max-width:1180px;padding-top:2rem;padding-bottom:3rem}
.hero{display:flex;align-items:center;gap:12px;margin-bottom:1.2rem}
.hero-icon{width:44px;height:44px;display:grid;place-items:center;border-radius:14px;background:#227d74;color:white;font-size:22px;box-shadow:0 7px 18px #227d742c}
.hero-name{font-family:Manrope,sans-serif;font-size:23px;font-weight:800;letter-spacing:-1px;color:var(--ink)}
.hero-tag{color:#43867c;font-size:10px;font-weight:700;letter-spacing:1.3px;text-transform:uppercase}
.hero-title{font-family:Manrope,sans-serif;color:var(--ink);font-size:clamp(27px,4vw,40px);line-height:1.15;letter-spacing:-1.5px;font-weight:800;margin:.4rem 0}
.hero-copy{color:var(--muted);font-size:14px;margin-bottom:1.4rem}
.card{background:white;border:1px solid var(--line);border-radius:16px;padding:20px 22px;box-shadow:0 5px 20px #1b3b2710;margin-bottom:15px}
.stButton>button[kind="primary"]{background:#227d74;border-color:#227d74;border-radius:9px;font-weight:700;min-height:42px}
[data-testid="stSidebar"]{background:#fbfcfa;border-right:1px solid var(--line)}
.source-label{font-size:11px;color:#5d7971}
.answer{line-height:1.7}
.small-muted{font-size:11px;color:#84918d;line-height:1.6}
</style>
""", unsafe_allow_html=True)

st.sidebar.markdown("## ✚ CareDocs")
st.sidebar.caption("Your medical document companion")
st.sidebar.markdown("---")
st.sidebar.markdown("### Privacy & AI")
api_key = setting("OPENAI_API_KEY").strip()
base_url = setting("OPENAI_BASE_URL", "https://api.openai.com/v1")
model = setting("OPENAI_MODEL", "gpt-4o-mini")
if api_key:
    st.sidebar.warning("Optional AI is enabled. Retrieved document excerpts and your question are sent to the configured external AI provider when you ask a question.")
    external_mode = True
else:
    st.sidebar.success("Local-only answer mode. Document contents are not sent to an AI provider.")
    external_mode = False
st.sidebar.markdown("---")
st.sidebar.markdown("### Your documents")
uploads = st.sidebar.file_uploader("Add PDF, TXT, or Markdown files", type=["pdf", "txt", "md", "markdown"], accept_multiple_files=True)
if st.sidebar.button("Clear session documents", use_container_width=True):
    for key in ("care_docs", "care_chunks", "care_messages", "care_errors"):
        st.session_state.pop(key, None)
    st.rerun()
st.sidebar.caption("Documents are kept in Streamlit session memory, not written to disk by this app.")

st.markdown('<div class="hero"><div class="hero-icon">✚</div><div class="hero-name">caredocs<span style="color:#227d74">.</span></div></div>', unsafe_allow_html=True)
st.markdown('<div class="hero-tag">PERSONAL MEDICAL DOCUMENT Q&A</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-title">Your documents, easier to understand.</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-copy">Ask questions about information in your own uploaded documents and review the source excerpts behind each answer.</div>', unsafe_allow_html=True)
st.error(f"**Medical safety:** {SAFETY_NOTICE}")
st.info("Only upload documents you are allowed to use. Do not upload another person's identifiable health information without their informed consent. CareDocs is not a diagnosis or treatment tool.")

# Rebuild only when upload set/content changes. Keep extracted text transient in session state.
if uploads is not None:
    signature = tuple((file.name, hashlib.sha256(file.getvalue()).hexdigest()) for file in uploads)
    if signature != st.session_state.get("care_doc_signature"):
        documents = []
        errors = []
        for uploaded in uploads:
            try:
                documents.append((uploaded.name, extract_document(uploaded.name, uploaded.getvalue())))
            except Exception as exc:
                errors.append(f"{uploaded.name}: {exc}")
        st.session_state.care_docs = documents
        st.session_state.care_errors = errors
        st.session_state.care_doc_signature = signature
        st.session_state.pop("care_chunks", None)
        st.session_state.pop("care_messages", None)

for message in st.session_state.get("care_errors", []):
    st.warning(message)

documents = st.session_state.get("care_docs", [])
if documents:
    if "care_chunks" not in st.session_state:
        st.session_state.care_chunks = chunk_documents(documents)
    chunks = st.session_state.care_chunks
    st.markdown(f"**{len(documents)} document(s)** indexed into **{len(chunks)} searchable passages**.")
    with st.expander("Documents in this session"):
        for name, text in documents:
            st.markdown(f"- **{name}** · {len(text):,} extracted characters")
else:
    chunks = []
    st.markdown('<div class="card"><h3>Start with your own documents</h3><p class="small-muted">Upload a visit summary, discharge instructions, lab report, or personal health notes. CareDocs includes no sample medical content and will not invent document citations.</p></div>', unsafe_allow_html=True)

if "care_messages" not in st.session_state:
    st.session_state.care_messages = [{
        "role": "assistant",
        "content": "Hello. Upload one or more of your own documents, then ask a question about what they say. I can summarize retrieved text, but I cannot diagnose, recommend treatment, or replace your clinician.",
        "sources": [],
    }]

chat_col, details_col = st.columns([1.5, 0.75], gap="large")
with chat_col:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Ask about your documents")
    st.caption("Answers are grounded in retrieved excerpts. If no relevant source is found, the assistant should say so.")
    for message in st.session_state.care_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"], unsafe_allow_html=False)
            if message.get("external"):
                st.caption("Generated by the configured external AI provider using retrieved excerpts.")
    question = st.chat_input("For example: What follow-up instructions are in my visit summary?", disabled=not chunks)
    if question:
        st.session_state.care_messages.append({"role": "user", "content": question, "sources": []})
        retriever = TfidfRetriever(chunks)
        retrieved = retriever.search(question, top_k=4)
        try:
            with st.spinner("Searching your session documents…"):
                answer, external_used = generate_answer(
                    question,
                    retrieved,
                    api_key=api_key or None,
                    base_url=base_url,
                    model=model,
                )
            st.session_state.care_messages.append({
                "role": "assistant",
                "content": answer,
                "sources": [(chunk.source, chunk.number, score) for chunk, score in retrieved],
                "external": external_used,
            })
        except Exception as exc:
            st.session_state.care_messages.append({
                "role": "assistant",
                "content": "The answer service is unavailable. I won't guess. Please try again or consult a healthcare professional.\n\n" + f"Technical detail: {type(exc).__name__}",
                "sources": [],
                "external": False,
            })
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

with details_col:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### How CareDocs works")
    st.markdown("1. Extract text from your upload in memory.\n2. Split text into overlapping passages.\n3. Retrieve passages similar to your question.\n4. Show cited excerpts locally, or optionally ask the configured model to answer from those passages.")
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Safety reminders")
    st.markdown("- Not for emergencies, diagnosis, or treatment decisions.\n- Check source passages; retrieved text can be incomplete.\n- Contact your clinician for interpretation.\n- For severe or life-threatening symptoms, call local emergency services.")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<p class="small-muted" style="text-align:center;margin-top:24px">CareDocs is a privacy-conscious demo, not certified health-record software. Retrieved sources are user-provided and are not independently verified.</p>', unsafe_allow_html=True)

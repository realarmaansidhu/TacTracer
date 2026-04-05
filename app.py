import os
import re
from datetime import datetime
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

st.set_page_config(page_title="TacTracer", page_icon="🛡️", layout="wide")
load_dotenv()

KB_PATHS = [
    Path("initial_data") / "mitre_knowledge.txt",
    Path("mitre_knowledge.txt"),
]


# ── Log Validation Guardrails ──────────────────────────────────────────────

# Patterns that real network / IoT logs contain
_LOG_PATTERNS = [
    # Zeek conn.log header
    re.compile(r"#fields\s+ts\b", re.IGNORECASE),
    # IP addresses (v4)
    re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"),
    # Common protocols
    re.compile(r"\b(tcp|udp|icmp|http|https|ssh|telnet|dns|ssl|ftp|smtp|modbus|mqtt)\b", re.IGNORECASE),
    # Ports as standalone numbers typical in logs
    re.compile(r"\b\d{1,5}\b"),
    # Connection states (Zeek)
    re.compile(r"\b(S0|S1|SF|REJ|RSTO|RSTR|SH|SHR|OTH)\b"),
    # Syslog-style timestamps
    re.compile(r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}\b"),
    # Unix epoch timestamps
    re.compile(r"\b1[4-7]\d{8}\.\d+\b"),
    # Common log labels / severities
    re.compile(r"\b(alert|warning|critical|info|notice|error|drop|deny|allow|accept|reject)\b", re.IGNORECASE),
]

# Minimum thresholds to classify as a valid log file
_MIN_IP_MATCHES = 2        # at least 2 IP addresses
_MIN_PATTERN_TYPES = 3     # at least 3 different pattern categories must match


def validate_logs(text: str) -> tuple[bool, str]:
    """
    Check whether the uploaded text looks like network / IoT log data.
    Returns (is_valid, reason).
    """
    if len(text.strip()) < 30:
        return False, "File is too short to be a meaningful log file."

    matched_categories = 0
    for i, pat in enumerate(_LOG_PATTERNS):
        hits = pat.findall(text)
        # Category 3 (bare numbers) needs many hits to count
        if i == 3:
            if len(hits) >= 6:
                matched_categories += 1
        elif len(hits) >= 1:
            matched_categories += 1

    ip_count = len(_LOG_PATTERNS[1].findall(text))

    if ip_count < _MIN_IP_MATCHES:
        return False, (
            "No IP addresses detected. This does not appear to be a network log file. "
            "TacTracer only analyses network / IoT log data."
        )

    if matched_categories < _MIN_PATTERN_TYPES:
        return False, (
            "The file does not contain recognisable network log patterns "
            "(IP addresses, protocols, ports, timestamps, connection states). "
            "TacTracer only analyses network / IoT log data — not general text."
        )

    return True, ""


# ── Knowledge Base & RAG ───────────────────────────────────────────────────

def load_knowledge_base() -> str:
    for path in KB_PATHS:
        if path.exists():
            return path.read_text(encoding="utf-8")
    raise FileNotFoundError("mitre_knowledge.txt not found. Place it in the project root.")


@st.cache_resource(show_spinner="Building MITRE ATT&CK knowledge index...")
def build_vector_store() -> FAISS:
    """
    Chunk and embed the MITRE knowledge base into an in-memory FAISS index.
    Cached by Streamlit — runs exactly once per session. No external DB needed.
    """
    kb_text = load_knowledge_base()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=80,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    docs = [
        Document(page_content=chunk.strip(), metadata={"chunk_id": i})
        for i, chunk in enumerate(splitter.split_text(kb_text))
        if chunk.strip()
    ]
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(docs, embeddings)


def extract_rag_query(logs: str) -> str:
    """
    Build a focused semantic query from log content.
    Extracts attack labels, destination ports, and protocols so the embedding
    model homes in on the right MITRE techniques instead of being drowned by
    raw tab-separated noise.
    """
    labels = re.findall(r'\t([A-Za-z][A-Za-z0-9_]+)\s*$', logs, re.MULTILINE)
    ports  = re.findall(r'\b(22|23|80|81|443|2323|8000|8080)\b', logs)
    protos = re.findall(r'\b(tcp|udp|http|https|ssh|telnet|ssl|dns|ftp)\b', logs, re.IGNORECASE)

    parts = []
    if labels:
        parts.append("Attack labels: " + " ".join(dict.fromkeys(labels)))
    if ports:
        parts.append("Ports: " + " ".join(dict.fromkeys(ports)))
    if protos:
        parts.append("Protocols: " + " ".join(dict.fromkeys(p.lower() for p in protos)))

    return " | ".join(parts) if parts else logs[:600]


def retrieve_mitre_context(logs: str, top_k: int = 5) -> list[Document]:
    """Semantic similarity search over the in-memory FAISS MITRE index."""
    query = extract_rag_query(logs)
    return build_vector_store().similarity_search(query, k=top_k)


def save_report(report_text: str) -> Path:
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = reports_dir / f"Incident_Report_{timestamp}.txt"
    path.write_text(report_text, encoding="utf-8")
    return path


PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an expert IoT Security Incident Responder. "
        "Analyse the provided network logs using the MITRE ATT&CK technique chunks retrieved below. "
        "Ground every finding in the retrieved techniques first; use your own security knowledge only "
        "to fill gaps, and clearly label such inferences as 'Analyst Inference'. "
        "Return a concise professional report with exactly these sections:\n"
        "1) Executive Summary\n"
        "2) Key Findings\n"
        "3) MITRE ATT&CK Mapping  (Technique ID | Technique Name | Evidence from logs)\n"
        "4) Risk Assessment\n"
        "5) Recommended Remediation Actions",
    ),
    (
        "user",
        "Retrieved MITRE ATT&CK Context:\n{context}\n\n"
        "Network Logs:\n{logs}\n\n"
        "Generate the incident response report.",
    ),
])


# ── UI ────────────────────────────────────────────────────────────────────────

st.title("🛡️ TacTracer: Autonomous IoT Threat Hunter")
st.markdown(
    "Upload Zeek / IoT network logs and let TacTracer cross-reference them against "
    "a MITRE ATT&CK knowledge base to produce an incident response report."
)

uploaded_files = st.file_uploader(
    "Upload IoT Log Files (.txt / .log / .csv)",
    type=["txt", "log", "csv"],
    accept_multiple_files=True,
)

if not uploaded_files:
    st.info("Upload one or more IoT / network log files to begin analysis.")
else:
    # ── Guardrail: validate every uploaded file ──
    combined_logs = ""
    rejected = []
    for f in uploaded_files:
        raw = f.getvalue().decode("utf-8", errors="replace")
        valid, reason = validate_logs(raw)
        if not valid:
            rejected.append((f.name, reason))
        else:
            combined_logs += f"\n--- File: {f.name} ---\n{raw}\n"

    if rejected:
        for fname, reason in rejected:
            st.error(f"**{fname}** rejected — {reason}")

    if not combined_logs.strip():
        st.warning("No valid log files to analyse. Please upload network / IoT log data.")
    else:
        st.subheader("Ingested Logs")
        st.code(combined_logs, language="text")

        if st.button("Run RAG Analysis"):
            with st.spinner("Retrieving MITRE context and generating report..."):
                try:
                    # Step 1 — Retrieve: semantic search over embedded MITRE KB
                    retrieved_docs = retrieve_mitre_context(combined_logs)
                    rag_context = "\n\n".join(doc.page_content for doc in retrieved_docs)

                    # Step 2 — Augment + Generate: LLM conditioned on retrieved chunks
                    if not os.getenv("GROQ_API_KEY"):
                        raise ValueError("GROQ_API_KEY missing from .env")
                    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.2)
                    response = (PROMPT | llm).invoke({"context": rag_context, "logs": combined_logs})

                    report_text = response.content if isinstance(response.content, str) else str(response.content)
                    report_path = save_report(report_text)

                    st.success("Analysis complete.")
                    st.subheader("Forensic Report")
                    st.markdown(report_text)
                    st.caption(f"Saved to: {report_path}")

                    with st.expander("Retrieved RAG Context (MITRE chunks used by the LLM)"):
                        for i, doc in enumerate(retrieved_docs, 1):
                            st.markdown(f"**Chunk {i}**")
                            st.write(doc.page_content)

                    st.download_button(
                        label="Download Report (.txt)",
                        data=report_text,
                        file_name=report_path.name,
                        mime="text/plain",
                    )

                except Exception as exc:
                    st.error(f"Analysis failed: {exc}")

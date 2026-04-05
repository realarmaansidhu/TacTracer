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

# ── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Global font & background */
    .stApp {
        background: linear-gradient(160deg, #0a0e17 0%, #0d1525 40%, #111d2e 100%);
    }

    /* Header area */
    .hero-title {
        font-size: 2.6rem;
        font-weight: 800;
        background: linear-gradient(135deg, #00d4ff, #7b2ff7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .hero-subtitle {
        color: #8899aa;
        font-size: 1.05rem;
        margin-top: 4px;
        margin-bottom: 28px;
    }

    /* Stat cards row */
    .stat-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 18px 20px;
        text-align: center;
    }
    .stat-value {
        font-size: 1.7rem;
        font-weight: 700;
        color: #00d4ff;
    }
    .stat-label {
        font-size: 0.82rem;
        color: #667788;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 2px;
    }

    /* Section headers */
    .section-header {
        font-size: 1.15rem;
        font-weight: 600;
        color: #c0ccdd;
        border-bottom: 2px solid rgba(0,212,255,0.25);
        padding-bottom: 8px;
        margin-top: 28px;
        margin-bottom: 14px;
    }

    /* File pills */
    .file-pill {
        display: inline-block;
        background: rgba(0,212,255,0.10);
        border: 1px solid rgba(0,212,255,0.25);
        color: #00d4ff;
        border-radius: 20px;
        padding: 5px 14px;
        margin: 3px 4px;
        font-size: 0.85rem;
        font-family: monospace;
    }

    /* Log viewer */
    .log-viewer {
        background: #0b1120;
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 10px;
        padding: 16px;
        font-family: 'SF Mono', 'Fira Code', monospace;
        font-size: 0.78rem;
        color: #a0b0c0;
        max-height: 320px;
        overflow-y: auto;
        line-height: 1.6;
    }

    /* Report container */
    .report-container {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 14px;
        padding: 28px 32px;
        margin-top: 16px;
    }

    /* RAG chunk cards */
    .rag-chunk {
        background: rgba(123,47,247,0.06);
        border-left: 3px solid #7b2ff7;
        border-radius: 0 8px 8px 0;
        padding: 12px 16px;
        margin-bottom: 10px;
        font-size: 0.88rem;
        color: #b0b8cc;
    }

    /* Rejected file warning */
    .rejected-file {
        background: rgba(255, 75, 75, 0.08);
        border: 1px solid rgba(255, 75, 75, 0.25);
        border-radius: 10px;
        padding: 12px 16px;
        margin-bottom: 8px;
        color: #ff6b6b;
    }

    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #00d4ff, #7b2ff7) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 12px 32px !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        transition: all 0.3s ease !important;
    }
    .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 20px rgba(0,212,255,0.3) !important;
    }

    /* Download button */
    .stDownloadButton > button {
        background: rgba(255,255,255,0.06) !important;
        color: #00d4ff !important;
        border: 1px solid rgba(0,212,255,0.3) !important;
        border-radius: 10px !important;
    }

    /* Hide default streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Spinner */
    .stSpinner > div > div {
        border-top-color: #00d4ff !important;
    }

    /* ── Global text contrast fixes ── */
    /* Main content text — bright white */
    .stMarkdown p, .stMarkdown li, .stMarkdown td, .stMarkdown th,
    .stMarkdown span, .stMarkdown ol, .stMarkdown ul {
        color: #ffffff !important;
    }
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3,
    .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {
        color: #ffffff !important;
    }
    .stMarkdown strong, .stMarkdown b {
        color: #ffffff !important;
    }

    /* Expander header */
    [data-testid="stExpander"] summary span,
    [data-testid="stExpander"] summary p {
        color: #ffffff !important;
    }
    /* Expander body text */
    [data-testid="stExpander"] [data-testid="stMarkdownContainer"] p {
        color: #ffffff !important;
    }

    /* File uploader — everything outside the dropzone gets light text */
    [data-testid="stFileUploader"] * {
        color: #c0d0e0 !important;
    }
    [data-testid="stFileUploader"] svg,
    [data-testid="stFileUploader"] path {
        color: #c0d0e0 !important;
        fill: #c0d0e0 !important;
        stroke: #c0d0e0 !important;
    }
    /* Dropzone itself — revert to its own dark-on-light styling */
    [data-testid="stFileUploaderDropzone"],
    [data-testid="stFileUploaderDropzone"] * {
        color: unset !important;
    }
    [data-testid="stFileUploaderDropzone"] svg,
    [data-testid="stFileUploaderDropzone"] path {
        color: unset !important;
        fill: unset !important;
        stroke: unset !important;
    }

    /* Info / warning / success / error boxes — keep native colors */
    .stAlert p, .stAlert span {
        color: inherit !important;
    }

    /* Caption text */
    .stCaption p, .stCaption span {
        color: #aabbcc !important;
    }
</style>
""", unsafe_allow_html=True)


KB_PATHS = [
    Path("initial_data") / "mitre_knowledge.txt",
    Path("mitre_knowledge.txt"),
]


# ── Log Validation Guardrails ──────────────────────────────────────────────

_LOG_PATTERNS = [
    re.compile(r"#fields\s+ts\b", re.IGNORECASE),
    re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"),
    re.compile(r"\b(tcp|udp|icmp|http|https|ssh|telnet|dns|ssl|ftp|smtp|modbus|mqtt)\b", re.IGNORECASE),
    re.compile(r"\b\d{1,5}\b"),
    re.compile(r"\b(S0|S1|SF|REJ|RSTO|RSTR|SH|SHR|OTH)\b"),
    re.compile(r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}\b"),
    re.compile(r"\b1[4-7]\d{8}\.\d+\b"),
    re.compile(r"\b(alert|warning|critical|info|notice|error|drop|deny|allow|accept|reject)\b", re.IGNORECASE),
]

_MIN_IP_MATCHES = 2
_MIN_PATTERN_TYPES = 3


def validate_logs(text: str) -> tuple[bool, str]:
    if len(text.strip()) < 30:
        return False, "File is too short to be a meaningful log file."

    matched_categories = 0
    for i, pat in enumerate(_LOG_PATTERNS):
        hits = pat.findall(text)
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


def count_log_lines(text: str) -> int:
    return sum(1 for line in text.strip().splitlines() if line.strip() and not line.startswith("#"))


def extract_unique_ips(text: str) -> set[str]:
    return set(re.findall(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b", text))


def extract_threat_labels(text: str) -> list[str]:
    """Extract labels from IoT-23 Zeek conn.log format (last two columns: label, detailed-label)."""
    labels = []
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        cols = line.split("\t")
        # IoT-23 format: label is second-to-last, detailed-label is last
        if len(cols) >= 2:
            label = cols[-2].strip() if len(cols) >= 22 else cols[-1].strip()
            detailed = cols[-1].strip() if len(cols) >= 22 else ""
            # Prefer detailed-label if available and not "-"
            tag = detailed if detailed and detailed != "-" else label
            if tag and tag != "-":
                labels.append(tag)
    return list(dict.fromkeys(labels))


# ── Knowledge Base & RAG ───────────────────────────────────────────────────

def load_knowledge_base() -> str:
    for path in KB_PATHS:
        if path.exists():
            return path.read_text(encoding="utf-8")
    raise FileNotFoundError("mitre_knowledge.txt not found. Place it in the project root.")


@st.cache_resource(show_spinner="Building MITRE ATT&CK knowledge index...")
def build_vector_store() -> FAISS:
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
    labels = extract_threat_labels(logs)
    ports  = re.findall(r'\b(22|23|80|81|443|2323|8000|8080|8443)\b', logs)
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


# ── UI ─────────────────────────────────────────────────────────────────────

# Hero header
st.markdown('<p class="hero-title">TacTracer</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="hero-subtitle">'
    'Autonomous IoT Threat Hunter — Upload network logs, get MITRE ATT&CK-grounded forensic reports'
    '</p>',
    unsafe_allow_html=True,
)

# File uploader
uploaded_files = st.file_uploader(
    "Upload IoT / Network Log Files",
    type=["txt", "log", "csv"],
    accept_multiple_files=True,
    label_visibility="collapsed",
)

if not uploaded_files:
    # Landing state
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            '<div class="stat-card">'
            '<div class="stat-value">RAG</div>'
            '<div class="stat-label">MITRE ATT&CK Knowledge Base</div>'
            '</div>',
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            '<div class="stat-card">'
            '<div class="stat-value">FAISS</div>'
            '<div class="stat-label">In-Memory Vector Search</div>'
            '</div>',
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            '<div class="stat-card">'
            '<div class="stat-value">LLM</div>'
            '<div class="stat-label">AI-Powered Analysis</div>'
            '</div>',
            unsafe_allow_html=True,
        )

    st.markdown("")
    st.info("Upload **.txt**, **.log**, or **.csv** network log files to begin threat analysis.")

else:
    # ── Validate uploads ──
    combined_logs = ""
    rejected = []
    accepted_files = []

    for f in uploaded_files:
        raw = f.getvalue().decode("utf-8", errors="replace")
        valid, reason = validate_logs(raw)
        if not valid:
            rejected.append((f.name, reason))
        else:
            accepted_files.append(f.name)
            combined_logs += f"\n--- File: {f.name} ---\n{raw}\n"

    # Show rejected files
    if rejected:
        for fname, reason in rejected:
            st.markdown(
                f'<div class="rejected-file"><strong>{fname}</strong> — {reason}</div>',
                unsafe_allow_html=True,
            )

    if not combined_logs.strip():
        st.warning("No valid log files to analyse. Please upload network / IoT log data.")
    else:
        # ── Stats bar ──
        total_lines = count_log_lines(combined_logs)
        unique_ips = extract_unique_ips(combined_logs)
        threat_labels = extract_threat_labels(combined_logs)
        malicious_labels = [l for l in threat_labels if l.lower() != "benign"]

        st.markdown("---")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(
                f'<div class="stat-card"><div class="stat-value">{len(accepted_files)}</div>'
                f'<div class="stat-label">Files Ingested</div></div>',
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                f'<div class="stat-card"><div class="stat-value">{total_lines}</div>'
                f'<div class="stat-label">Log Entries</div></div>',
                unsafe_allow_html=True,
            )
        with c3:
            st.markdown(
                f'<div class="stat-card"><div class="stat-value">{len(unique_ips)}</div>'
                f'<div class="stat-label">Unique IPs</div></div>',
                unsafe_allow_html=True,
            )
        with c4:
            st.markdown(
                f'<div class="stat-card"><div class="stat-value">{len(malicious_labels)}</div>'
                f'<div class="stat-label">Threat Types</div></div>',
                unsafe_allow_html=True,
            )

        # ── Accepted file pills ──
        st.markdown("")
        pills_html = "".join(f'<span class="file-pill">{fn}</span>' for fn in accepted_files)
        st.markdown(pills_html, unsafe_allow_html=True)

        # ── Log preview ──
        st.markdown('<div class="section-header">Log Preview</div>', unsafe_allow_html=True)
        preview_lines = combined_logs.strip().split("\n")[:25]
        preview_text = "\n".join(preview_lines)
        if len(combined_logs.strip().split("\n")) > 25:
            preview_text += f"\n\n... ({total_lines} total entries across {len(accepted_files)} files)"
        st.markdown(
            f'<div class="log-viewer"><pre>{preview_text}</pre></div>',
            unsafe_allow_html=True,
        )

        # ── Analysis button ──
        st.markdown("")
        if st.button("Analyse Threats", use_container_width=True):
            with st.spinner("Retrieving MITRE ATT&CK context and generating forensic report..."):
                try:
                    retrieved_docs = retrieve_mitre_context(combined_logs)
                    rag_context = "\n\n".join(doc.page_content for doc in retrieved_docs)

                    if not os.getenv("GROQ_API_KEY"):
                        raise ValueError("GROQ_API_KEY missing from .env")
                    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.2)
                    response = (PROMPT | llm).invoke({"context": rag_context, "logs": combined_logs})

                    report_text = response.content if isinstance(response.content, str) else str(response.content)
                    report_path = save_report(report_text)

                    # ── Report display ──
                    st.markdown('<div class="section-header">Forensic Report</div>', unsafe_allow_html=True)
                    st.markdown(report_text)

                    st.caption(f"Report saved to `{report_path}`")

                    # Download button
                    st.download_button(
                        label="Download Report",
                        data=report_text,
                        file_name=report_path.name,
                        mime="text/plain",
                    )

                    # ── RAG context expander ──
                    with st.expander("View Retrieved MITRE ATT&CK Context (RAG Chunks)"):
                        for i, doc in enumerate(retrieved_docs, 1):
                            st.markdown(
                                f'<div class="rag-chunk"><strong>Chunk {i}</strong><br>{doc.page_content}</div>',
                                unsafe_allow_html=True,
                            )

                except Exception as exc:
                    st.error(f"Analysis failed: {exc}")

import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import streamlit as st
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

st.set_page_config(page_title="TacTracer", page_icon="🔬", layout="wide")
load_dotenv()


def resolve_config_value(key: str) -> Tuple[Optional[str], Optional[str]]:
    """Resolve config from Streamlit secrets first, then local environment."""
    # 1) Flat key in secrets.toml: GROQ_API_KEY = "..."
    try:
        if key in st.secrets:
            value = str(st.secrets[key]).strip()
            if value:
                return value, "streamlit-secrets"
    except Exception:
        pass

    # 2) Nested key in secrets.toml: [api_keys] GROQ_API_KEY = "..."
    try:
        if "api_keys" in st.secrets and key in st.secrets["api_keys"]:
            value = str(st.secrets["api_keys"][key]).strip()
            if value:
                return value, "streamlit-secrets.api_keys"
    except Exception:
        pass

    # 3) Local .env / OS environment
    value = os.getenv(key, "").strip()
    if value:
        return value, "local-env"

    return None, None


def ensure_runtime_keys() -> Dict[str, str]:
    """Ensure required keys exist and normalize them into process env for SDKs."""
    resolved: Dict[str, str] = {}

    groq_key, source = resolve_config_value("GROQ_API_KEY")
    if not groq_key:
        raise ValueError(
            "Missing GROQ_API_KEY. Add it to Streamlit secrets (preferred) or local .env."
        )

    # Normalize into environment so dependent clients read a consistent key location.
    os.environ["GROQ_API_KEY"] = groq_key
    resolved["GROQ_API_KEY"] = source or "unknown"
    return resolved

# ── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* ═══ BASE ═══ */
    .stApp {
        background: linear-gradient(160deg, #0a0e17 0%, #0d1525 40%, #111d2e 100%);
    }

    /* Hide streamlit chrome */
    #MainMenu, footer, header { visibility: hidden; }

    /* ═══ HERO ═══ */
    p.hero-title {
        font-size: 3.4rem !important;
        font-weight: 800 !important;
        background: linear-gradient(135deg, #00d4ff, #7b2ff7) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        text-align: center !important;
        margin-bottom: 0 !important;
        line-height: 1.2 !important;
    }
    p.hero-subtitle {
        color: #b0bdd0 !important;
        font-size: 1.25rem !important;
        text-align: center !important;
        margin-top: 4px !important;
        margin-bottom: 32px !important;
        font-weight: 400 !important;
    }

    /* ═══ STAT CARDS ═══ */
    .stat-card {
        background: rgba(0,20,50,0.7) !important;
        border: 1px solid rgba(0,212,255,0.15) !important;
        border-radius: 12px; padding: 22px 20px; text-align: center;
    }
    .stat-card .stat-value {
        font-size: 1.8rem !important; font-weight: 700 !important; color: #00d4ff !important;
    }
    .stat-card .stat-label {
        font-size: 0.9rem !important; color: #8899bb !important;
        text-transform: uppercase; letter-spacing: 0.5px; margin-top: 4px;
    }

    /* ═══ SECTION HEADERS ═══ */
    .section-header {
        font-size: 1.4rem !important; font-weight: 700 !important; color: #00d4ff !important;
        border-bottom: 2px solid rgba(0,212,255,0.25) !important;
        padding-bottom: 10px !important; margin-top: 36px !important; margin-bottom: 18px !important;
    }

    /* ═══ FILE PILLS ═══ */
    .file-pill {
        display: inline-block; background: rgba(0,212,255,0.10);
        border: 1px solid rgba(0,212,255,0.25); color: #00d4ff !important;
        border-radius: 20px; padding: 6px 16px; margin: 4px 5px;
        font-size: 0.95rem; font-family: monospace;
    }

    /* ═══ LOG VIEWER ═══ */
    .log-viewer {
        background: #0b1120; border: 1px solid rgba(255,255,255,0.06);
        border-radius: 10px; padding: 18px;
        font-family: 'SF Mono','Fira Code',monospace; font-size: 0.85rem;
        color: #b0c0d0; max-height: 350px; overflow-y: auto; line-height: 1.7;
    }

    /* ═══ RAG CHUNKS ═══ */
    .rag-chunk {
        background: rgba(123,47,247,0.06); border-left: 3px solid #7b2ff7;
        border-radius: 0 8px 8px 0; padding: 16px 20px; margin-bottom: 14px;
        font-size: 1.0rem !important; color: #c0c8dd !important; line-height: 1.7;
    }

    /* ═══ REJECTED FILE ═══ */
    .rejected-file {
        background: rgba(255,75,75,0.08); border: 1px solid rgba(255,75,75,0.25);
        border-radius: 10px; padding: 12px 16px; margin-bottom: 8px; color: #ff6b6b;
    }

    /* ═══ BUTTONS ═══ */
    .stButton > button {
        background: linear-gradient(135deg, #00d4ff, #7b2ff7) !important;
        color: white !important; border: none !important; border-radius: 10px !important;
        padding: 14px 32px !important; font-weight: 600 !important; font-size: 1.05rem !important;
    }
    .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 20px rgba(0,212,255,0.3) !important;
    }
    .stDownloadButton > button {
        background: rgba(255,255,255,0.06) !important; color: #00d4ff !important;
        border: 1px solid rgba(0,212,255,0.3) !important; border-radius: 10px !important;
    }

    /* ═══ SPINNER ═══ */
    /* The circle */
    .stApp div[data-testid="stSpinner"] > div > i,
    .stApp div[data-testid="stSpinner"] > div > div,
    .stApp .stSpinner > div > div,
    .stApp .stSpinner > div > i {
        border-color: rgba(0,212,255,0.2) !important;
        border-top-color: #00d4ff !important;
    }
    /* The text */
    .stApp div[data-testid="stSpinner"],
    .stApp div[data-testid="stSpinner"] > div,
    .stApp div[data-testid="stSpinner"] > div > span,
    .stApp div[data-testid="stSpinner"] span,
    .stApp .stSpinner,
    .stApp .stSpinner span {
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
    }

    /* ═══ REPORT TYPOGRAPHY ═══ */
    /* Body text — bumped to 1.1rem */
    [data-testid="stMarkdownContainer"] p {
        color: #e0e6f0 !important;
        font-size: 1.1rem !important;
        line-height: 1.8 !important;
    }
    /* H1 */
    [data-testid="stMarkdownContainer"] h1 {
        color: #ffffff !important;
        font-size: 2rem !important;
        font-weight: 700 !important;
        margin-top: 40px !important;
        margin-bottom: 16px !important;
        padding-bottom: 12px !important;
        border-bottom: 2px solid rgba(0,212,255,0.25) !important;
    }
    /* H2 — report section titles */
    [data-testid="stMarkdownContainer"] h2 {
        color: #00d4ff !important;
        font-size: 1.6rem !important;
        font-weight: 700 !important;
        margin-top: 34px !important;
        margin-bottom: 14px !important;
        padding-bottom: 6px !important;
        border-bottom: 1px solid rgba(0,212,255,0.15) !important;
    }
    /* H3 */
    [data-testid="stMarkdownContainer"] h3 {
        color: #7bb8ff !important;
        font-size: 1.3rem !important;
        font-weight: 600 !important;
        margin-top: 28px !important;
        margin-bottom: 12px !important;
    }
    /* H4-H6 */
    [data-testid="stMarkdownContainer"] h4,
    [data-testid="stMarkdownContainer"] h5,
    [data-testid="stMarkdownContainer"] h6 {
        color: #a0c4e8 !important;
        font-size: 1.15rem !important;
        font-weight: 600 !important;
        margin-top: 20px !important;
        margin-bottom: 10px !important;
    }
    /* Bold */
    [data-testid="stMarkdownContainer"] strong,
    [data-testid="stMarkdownContainer"] b {
        color: #ffffff !important;
        font-size: inherit !important;
    }
    /* Lists — bigger */
    [data-testid="stMarkdownContainer"] ul,
    [data-testid="stMarkdownContainer"] ol {
        color: #e0e6f0 !important;
        padding-left: 30px !important;
        margin-bottom: 16px !important;
    }
    [data-testid="stMarkdownContainer"] li {
        color: #e0e6f0 !important;
        font-size: 1.05rem !important;
        line-height: 1.8 !important;
        margin-bottom: 10px !important;
    }
    /* Tables — bigger */
    [data-testid="stMarkdownContainer"] table {
        width: 100% !important; border-collapse: collapse !important; margin: 16px 0 !important;
    }
    [data-testid="stMarkdownContainer"] th {
        background: rgba(0,212,255,0.1) !important; color: #ffffff !important;
        font-weight: 600 !important; text-align: left !important; padding: 12px 16px !important;
        border-bottom: 2px solid rgba(0,212,255,0.25) !important; font-size: 1.0rem !important;
    }
    [data-testid="stMarkdownContainer"] td {
        color: #d0d8e8 !important; padding: 10px 16px !important;
        border-bottom: 1px solid rgba(255,255,255,0.06) !important; font-size: 1.0rem !important;
    }

    /* ═══ HERO OVERRIDES — must beat [data-testid] p ═══ */
    [data-testid="stMarkdownContainer"] p.hero-title {
        font-size: 3.4rem !important;
        font-weight: 800 !important;
        background: linear-gradient(135deg, #00d4ff, #7b2ff7) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        text-align: center !important;
        margin-bottom: 0 !important;
        line-height: 1.2 !important;
        border: none !important;
    }
    [data-testid="stMarkdownContainer"] p.hero-subtitle {
        color: #b0bdd0 !important;
        -webkit-text-fill-color: #b0bdd0 !important;
        font-size: 1.15rem !important;
        text-align: center !important;
        margin-top: 4px !important;
        margin-bottom: 28px !important;
        font-weight: 400 !important;
        border: none !important;
    }
    /* Stat card inner elements — beat global p/span */
    [data-testid="stMarkdownContainer"] .stat-card .stat-value {
        font-size: 1.8rem !important; color: #00d4ff !important;
        -webkit-text-fill-color: #00d4ff !important; line-height: 1.3 !important;
    }
    [data-testid="stMarkdownContainer"] .stat-card .stat-label {
        font-size: 0.8rem !important; color: #8899bb !important;
        -webkit-text-fill-color: #8899bb !important;
    }
    /* Section header — beat global p */
    [data-testid="stMarkdownContainer"] .section-header {
        font-size: 1.6rem !important; font-weight: 700 !important; color: #00d4ff !important;
        -webkit-text-fill-color: #00d4ff !important;
        border-bottom: 2px solid rgba(0,212,255,0.25) !important;
        padding-bottom: 12px !important; margin-top: 40px !important; margin-bottom: 20px !important;
    }
    /* File pill — beat global span */
    [data-testid="stMarkdownContainer"] .file-pill {
        font-size: 0.85rem !important; color: #00d4ff !important;
        -webkit-text-fill-color: #00d4ff !important;
    }

    /* ═══ EXPANDER ═══ */
    [data-testid="stExpander"] summary span,
    [data-testid="stExpander"] summary p {
        color: #ffffff !important;
    }

    /* ═══ FILE UPLOADER — NUCLEAR: force ALL text visible ═══ */
    [data-testid="stFileUploader"],
    [data-testid="stFileUploader"] *,
    [data-testid="stFileUploader"] div,
    [data-testid="stFileUploader"] span,
    [data-testid="stFileUploader"] p,
    [data-testid="stFileUploader"] small,
    [data-testid="stFileUploader"] label,
    [data-testid="stFileUploader"] section,
    [data-testid="stFileUploaderDropzone"],
    [data-testid="stFileUploaderDropzone"] *,
    [data-testid="stFileUploaderDropzone"] div,
    [data-testid="stFileUploaderDropzone"] span,
    [data-testid="stFileUploaderDropzone"] p,
    [data-testid="stFileUploaderDropzone"] small,
    [data-testid="stFileUploaderDropzoneInstructions"],
    [data-testid="stFileUploaderDropzoneInstructions"] * {
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
    }
    /* Dropzone secondary/limit text — slightly dimmer */
    [data-testid="stFileUploaderDropzone"] small,
    [data-testid="stFileUploaderDropzone"] div[data-testid="stFileUploaderDropzoneInstructions"] div:last-child,
    [data-testid="stFileUploaderDropzone"] div[data-testid="stFileUploaderDropzoneInstructions"] span:last-child {
        color: #99aabb !important;
        -webkit-text-fill-color: #99aabb !important;
    }
    /* Dropzone background */
    [data-testid="stFileUploaderDropzone"] {
        background: rgba(255,255,255,0.03) !important;
        border: 1px dashed rgba(0,212,255,0.3) !important;
    }
    /* SVGs everywhere in uploader */
    [data-testid="stFileUploader"] svg,
    [data-testid="stFileUploaderDropzone"] svg {
        color: #ffffff !important;
    }
    [data-testid="stFileUploader"] svg line,
    [data-testid="stFileUploader"] svg polyline,
    [data-testid="stFileUploader"] svg path,
    [data-testid="stFileUploader"] svg circle,
    [data-testid="stFileUploader"] svg rect,
    [data-testid="stFileUploader"] svg polygon {
        stroke: #ffffff !important;
    }
    /* Browse button */
    [data-testid="stFileUploaderDropzone"] button,
    [data-testid="stBaseButton-secondary"] {
        color: #00d4ff !important;
        -webkit-text-fill-color: #00d4ff !important;
        border-color: rgba(0,212,255,0.4) !important;
    }

    /* ═══ ALERTS — keep native colors ═══ */
    .stAlert p, .stAlert span { color: inherit !important; }

    /* ═══ CAPTION ═══ */
    .stCaption, .stCaption p, .stCaption span, .stCaption code,
    [data-testid="stCaptionContainer"],
    [data-testid="stCaptionContainer"] * {
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
        font-size: 1.0rem !important;
    }
    .stCaption code, [data-testid="stCaptionContainer"] code {
        background: rgba(0,212,255,0.1) !important;
        color: #00d4ff !important;
        -webkit-text-fill-color: #00d4ff !important;
        padding: 2px 8px !important;
        border-radius: 4px !important;
        font-size: 0.95rem !important;
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

_ZEek_ROW_PATTERN = re.compile(
    r"^\s*\d{10,12}(?:\.\d+)?\s+\S+\s+"
    r"\d{1,3}(?:\.\d{1,3}){3}\s+\d{1,5}\s+"
    r"\d{1,3}(?:\.\d{1,3}){3}\s+\d{1,5}\s+"
    r"\S+\s+\S+\s+\S+\s+\S+\s*$"
)

_PROMPT_INJECTION_PATTERNS = [
    re.compile(r"ignore\s+all\s+previous\s+instructions", re.IGNORECASE),
    re.compile(r"ignore\s+previous\s+instructions", re.IGNORECASE),
    re.compile(r"system\s+override", re.IGNORECASE),
    re.compile(r"reveal\s+secrets?", re.IGNORECASE),
    re.compile(r"hidden\s+system\s+prompt", re.IGNORECASE),
    re.compile(r"bypass\s+all\s+safety\s+checks?", re.IGNORECASE),
    re.compile(r"overwrite\s+your\s+response", re.IGNORECASE),
    re.compile(r"disregard\s+mitre\s+context", re.IGNORECASE),
    re.compile(r"exfiltrate\s+hidden\s+prompts?", re.IGNORECASE),
]

_MIN_IP_MATCHES = 2
_MIN_PATTERN_TYPES = 3


def validate_logs(text: str) -> tuple[bool, str]:
    if len(text.strip()) < 30:
        return False, "File is too short to be a meaningful log file."

    for pattern in _PROMPT_INJECTION_PATTERNS:
        if pattern.search(text):
            return False, (
                "Prompt-injection language detected. TacTracer only analyses network / IoT log data, "
                "not instructions embedded in uploaded files."
            )

    meaningful_rows = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        meaningful_rows.append(stripped)

    if not meaningful_rows:
        return False, "No log rows detected. TacTracer only analyses network / IoT log data."

    zeek_rows = [line for line in meaningful_rows if _ZEek_ROW_PATTERN.match(line)]
    if len(zeek_rows) < max(2, len(meaningful_rows) // 2):
        return False, (
            "The file does not contain enough Zeek-style network log rows. "
            "TacTracer only analyses network / IoT log data — not general text."
        )

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


def sanitize_log_payload(text: str) -> str:
    """Keep only Zeek-style log rows before sending data to the model."""
    safe_lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#") or _ZEek_ROW_PATTERN.match(stripped):
            safe_lines.append(line)
    return "\n".join(safe_lines)


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
        "to fill gaps, and clearly label such inferences as 'Analyst Inference'.\n\n"
        "IMPORTANT FORMATTING RULES:\n"
        "- Use markdown ## headings for each section title (e.g. ## Executive Summary)\n"
        "- Use bullet points or numbered lists for findings and actions\n"
        "- Use a markdown table with columns: Technique ID | Technique Name | Evidence for the MITRE mapping\n"
        "- Use **bold** for key terms and threat names\n\n"
        "Sections to include:\n"
        "## Executive Summary\n"
        "## Key Findings\n"
        "## MITRE ATT&CK Mapping\n"
        "## Risk Assessment\n"
        "## Recommended Remediation Actions",
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
        st.error(
            "Upload rejected: one or more files contained malicious or prompt-injection content. "
            "Remove the flagged file(s) and upload only clean network / IoT logs."
        )
        for fname, reason in rejected:
            st.markdown(
                f'<div class="rejected-file"><strong>{fname}</strong> — {reason}</div>',
                unsafe_allow_html=True,
            )
        st.stop()

    if not combined_logs.strip():
        st.warning("No valid log files to analyse. Please upload network / IoT log data.")
    else:
        safe_logs = sanitize_log_payload(combined_logs)

        # ── Stats bar ──
        total_lines = count_log_lines(safe_logs)
        unique_ips = extract_unique_ips(safe_logs)
        threat_labels = extract_threat_labels(safe_logs)
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
        pills_html = '<div style="text-align:center;">' + "".join(f'<span class="file-pill">{fn}</span>' for fn in accepted_files) + '</div>'
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
                    retrieved_docs = retrieve_mitre_context(safe_logs)
                    rag_context = "\n\n".join(doc.page_content for doc in retrieved_docs)

                    key_sources = ensure_runtime_keys()
                    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.2)
                    response = (PROMPT | llm).invoke({"context": rag_context, "logs": safe_logs})

                    report_text = response.content if isinstance(response.content, str) else str(response.content)
                    report_path = save_report(report_text)

                    # ── Report display ──
                    st.markdown('<div class="section-header">Forensic Report</div>', unsafe_allow_html=True)
                    st.markdown(report_text)

                    st.caption(f"Report saved to `{report_path}`")
                    st.caption(f"GROQ key source: {key_sources['GROQ_API_KEY']}")

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
                            # Strip category headers (lines starting with =) from display
                            clean = "\n".join(
                                line for line in doc.page_content.splitlines()
                                if not line.strip().startswith("=")
                            ).strip()
                            st.markdown(
                                f'<div class="rag-chunk"><strong>Chunk {i}</strong><br>{clean}</div>',
                                unsafe_allow_html=True,
                            )

                except Exception as exc:
                    st.error(f"Analysis failed: {exc}")

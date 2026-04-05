import os
import re
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import streamlit as st
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

st.set_page_config(page_title="TacTrace Agent", page_icon="🛡️", layout="wide")
load_dotenv()

KB_CANDIDATES = [
    Path("initial_data") / "mitre_knowledge.txt",
    Path("mitre_knowledge.txt"),
]


def load_knowledge_base() -> Tuple[str, str]:
    candidates = [
        Path("initial_data") / "mitre_knowledge.txt",
        Path("mitre_knowledge.txt"),
    ]
    for path in candidates:
        if path.exists():
            return path.read_text(encoding="utf-8"), str(path)
    return "Technique: T1059.004 - Unix Shell via Telnet (23/2323).", "built-in fallback"


def build_chunks(knowledge_base: str) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    pieces = splitter.split_text(knowledge_base)
    return [
        Document(page_content=chunk.strip(), metadata={"chunk_id": idx})
        for idx, chunk in enumerate(pieces)
        if chunk.strip()
    ]


@st.cache_resource(show_spinner=False)
def build_retriever(knowledge_base: str, top_k: int = 5):
    docs = build_chunks(knowledge_base)
    if not docs:
        return None

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(docs, embeddings)
    return vector_store.as_retriever(search_kwargs={"k": top_k})


def retrieve_relevant_context_with_keywords(logs: str, knowledge_base: str, top_k: int = 5) -> str:
    chunks = [c.strip() for c in knowledge_base.split("\n\n") if c.strip()]
    if not chunks:
        return knowledge_base

    ports = set(re.findall(r"\b(?:port|dst_port|dport|sport|id\.resp_p)\D*(\d{1,5})\b", logs, flags=re.IGNORECASE))
    words = set(re.findall(r"[a-zA-Z_]{4,}", logs.lower()))

    scored = []
    for chunk in chunks:
        chunk_lower = chunk.lower()
        score = 0
        for p in ports:
            if p in chunk_lower:
                score += 5
        for w in words:
            if w in chunk_lower:
                score += 1
        scored.append((score, chunk))

    scored.sort(key=lambda x: x[0], reverse=True)
    selected = [chunk for score, chunk in scored[:top_k] if score > 0]
    if not selected:
        selected = [chunk for _, chunk in scored[:top_k]]
    return "\n\n".join(selected)


def retrieve_relevant_context(logs: str, knowledge_base: str, top_k: int = 5) -> Tuple[str, str]:
    try:
        retriever = build_retriever(knowledge_base, top_k=top_k)
        if retriever is None:
            raise ValueError("No retriever available")
        docs = retriever.invoke(logs)
        if not docs:
            raise ValueError("Retriever returned no documents")
        context = "\n\n".join(doc.page_content for doc in docs)
        return context, "semantic"
    except Exception:
        context = retrieve_relevant_context_with_keywords(logs, knowledge_base, top_k=top_k)
        return context, "keyword-fallback"

def save_report(report_text: str) -> Path:
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = reports_dir / f"Incident_Report_{timestamp}.txt"
    report_path.write_text(report_text, encoding="utf-8")
    return report_path

def normalize_content(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(str(item) for item in content)
    return str(content)

def build_llm() -> ChatGroq:
    if not os.getenv("GROQ_API_KEY"):
        raise ValueError("Missing GROQ_API_KEY in .env")
    return ChatGroq(model="llama-3.3-70b-versatile", temperature=0.2)

st.title("🛡️ TacTrace: Autonomous Multi-Agent Threat Hunter")
st.markdown("Upload Zeek network logs for automated forensic analysis and MITRE ATT&CK correlation.")

# UPGRADED: Batch File Uploader
uploaded_files = st.file_uploader("Upload IoT Log Files (.txt)", type="txt", accept_multiple_files=True)

if not uploaded_files:
    st.info("Upload one or more .txt IoT log files to start the RAG analysis.")

if uploaded_files:
    combined_logs = ""
    for file in uploaded_files:
        combined_logs += f"\n--- File: {file.name} ---\n"
        combined_logs += file.getvalue().decode("utf-8", errors="replace") + "\n"
        
    st.subheader("📡 Ingestion Agent: Batch Raw Logs")
    st.code(combined_logs)

    if st.button("Run Agentic Analysis"):
        with st.spinner("Agents are analyzing batch logs and querying the knowledge base..."):
            try:
                knowledge_base, kb_source = load_knowledge_base()
                retrieved_context, retrieval_mode = retrieve_relevant_context(combined_logs, knowledge_base)
                llm = build_llm()

                prompt = ChatPromptTemplate.from_messages(
                    [
                        (
                            "system",
                            "You are an expert IoT Security Incident Responder. Analyze the provided logs using MITRE ATT&CK context. "
                            "Use retrieved knowledge first. You may also use your own broader security reasoning where needed, "
                            "but clearly separate evidence-backed findings from assumptions. "
                            "Return a concise professional report with these sections only: "
                            "1) Executive Summary "
                            "2) Key Findings "
                            "3) MITRE ATT&CK Mapping (Technique ID + rationale) "
                            "4) Risk Assessment "
                            "5) Recommended Remediation Actions.",
                        ),
                        (
                            "user",
                            "Knowledge Base Context:\n{context}\n\nNetwork Logs:\n{logs}\n\nGenerate the incident response report now.",
                        ),
                    ]
                )

                chain = prompt | llm
                response = chain.invoke({"context": retrieved_context, "logs": combined_logs})
                report_text = normalize_content(response.content)
                report_path = save_report(report_text)

                st.success("Analysis complete.")
                st.caption(f"Knowledge base source: {kb_source}")
                st.caption(f"Retrieval mode: {retrieval_mode}")
                st.subheader("📄 Forensic Agent Report")
                st.write(report_text)
                st.caption(f"Saved report: {report_path}")

                with st.expander("View Retrieved RAG Context"):
                    st.write(retrieved_context)

                st.download_button(
                    label="Download Report (.txt)",
                    data=report_text,
                    file_name=report_path.name,
                    mime="text/plain",
                )
            except Exception as exc:
                st.error(f"Analysis failed: {exc}")
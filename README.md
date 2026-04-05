# TacTracer

**Autonomous IoT Threat Hunter** — Upload network logs, get MITRE ATT&CK-grounded forensic reports.

TacTracer is an AI-powered cybersecurity tool that analyses IoT and ICS network traffic logs using Retrieval-Augmented Generation (RAG). It maps suspicious activity to the MITRE ATT&CK framework and generates structured incident response reports — no manual threat hunting required.

Built for **INSE 6540 — IoT Security** at Concordia University.

---

## How It Works

```
Upload Logs ──> Guardrails ──> FAISS Vector Search ──> LLM Analysis ──> Forensic Report
                  (reject         (retrieve relevant       (Groq LLaMA 3.3     (structured
                   non-logs)       MITRE techniques)        70B Versatile)       markdown)
```

1. **Upload** — Drag and drop `.txt`, `.log`, or `.csv` network log files (Zeek conn.log format supported)
2. **Validate** — Input guardrails verify the files contain real network log data (IPs, protocols, ports, connection states). Random text, poems, or recipes are rejected.
3. **Retrieve** — A FAISS in-memory vector store indexes 50+ MITRE ATT&CK techniques covering IoT, ICS, and botnet threats. The most relevant technique chunks are retrieved based on detected threat labels, ports, and protocols.
4. **Analyse** — The retrieved context and raw logs are sent to Groq's LLaMA 3.3-70B model, which generates a forensic incident report grounded in MITRE ATT&CK mappings.
5. **Report** — A structured report is displayed with executive summary, key findings, MITRE technique table, risk assessment, and remediation actions. Reports are saved locally and downloadable.

---

## Tech Stack

| Component | Technology |
|---|---|
| **Frontend** | Streamlit (custom dark theme) |
| **Embeddings** | HuggingFace `sentence-transformers/all-MiniLM-L6-v2` |
| **Vector Store** | FAISS (in-memory, no external database) |
| **LLM** | Groq — `llama-3.3-70b-versatile` |
| **Framework** | LangChain (prompts, text splitters, document loaders) |
| **Knowledge Base** | Custom MITRE ATT&CK corpus (IoT/ICS/Botnet focus) |

---

## MITRE ATT&CK Coverage

The knowledge base covers the full attack lifecycle with 50+ techniques:

- **Initial Access** — T1190, T1133, T1078, T1199
- **Execution** — T1059.004, T1059.006, T1047
- **Persistence** — T1053.003, T1542.001, T1205
- **Credential Access** — T1110, T1110.001, T1110.003, T1040
- **Discovery** — T1046, T1018, T1082
- **Lateral Movement** — T1021, T1021.004, T1210
- **Command & Control** — T1071.001, T1071.004, T1573, T1095, T1571, T1132
- **Collection & Exfiltration** — T1119, T1041, T1048
- **Impact** — T1498, T1498.001, T1499, T1485, T1489
- **ICS-Specific** — T0883, T0855, T0859, T0814, T0879, T0816
- **Botnet Profiles** — Mirai, Hajime, Mozi, Bashlite, Torii, Dark Nexus

---

## Setup

### Prerequisites

- Python 3.9+
- A [Groq API key](https://console.groq.com)

### Installation

```bash
git clone https://github.com/realarmaansidhu/TacTracer.git
cd TacTracer

python3 -m venv environ
source environ/bin/activate

pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_groq_api_key_here
```

### Run

```bash
streamlit run app.py
```

The app launches at `http://localhost:8501`.

---

## Sample Logs

The `logs/` directory contains 5 sample log files in IoT-23 Zeek conn.log format with mixed threat scenarios:

| File | Threats Included |
|---|---|
| `iot_network_capture_01.txt` | Telnet scanning, C&C, DDoS, port scanning |
| `iot_network_capture_02.txt` | SSH brute force, C&C (Torii), horizontal scan |
| `iot_network_capture_03.txt` | Mirai-style scanning, C&C beaconing, DDoS |
| `iot_network_capture_04.txt` | Multi-protocol scanning, encrypted C&C, DDoS |
| `iot_network_capture_05.txt` | Telnet/SSH compromise, C&C, port scanning |

Each file contains 10 log entries with a mix of malicious and benign traffic, following the real IoT-23 Stratosphere dataset format (23 tab-separated fields with Zeek headers).

---

## Project Structure

```
TacTracer/
├── app.py                  # Main application (UI, guardrails, RAG pipeline, LLM)
├── mitre_knowledge.txt     # MITRE ATT&CK knowledge base (50+ techniques)
├── requirements.txt        # Python dependencies
├── .env                    # API keys (not tracked)
├── logs/                   # Sample IoT network log files
│   ├── iot_network_capture_01.txt
│   ├── ...
│   └── iot_network_capture_05.txt
└── reports/                # Generated incident reports (auto-created)
```

---

## Key Features

- **Input Guardrails** — Rejects non-network-log files using multi-pattern regex validation (IPs, protocols, ports, timestamps, connection states)
- **RAG Pipeline** — FAISS similarity search retrieves the most relevant MITRE ATT&CK techniques for each log set
- **Grounded Analysis** — LLM findings are anchored to retrieved techniques; independent inferences are clearly labelled
- **Batch Upload** — Analyse multiple log files simultaneously with aggregated statistics
- **Report Export** — Reports saved as timestamped `.txt` files and downloadable from the UI
- **Dark Theme UI** — Custom-styled Streamlit interface optimized for SOC analyst workflows

---

## License

MIT

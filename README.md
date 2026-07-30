# DocuMind — Streamlit Frontend

A production-style Streamlit UI for the DocuMind RAG platform. This
repository contains **frontend code only** — it imports and calls the
existing backend (`config.py`, `embeddings.py`, `ingestion.py`,
`retrieval.py`, `generation.py`, `evaluation.py`) and does not
reimplement any retrieval, generation, guardrail, or evaluation logic.

---

## Table of contents

- [DocuMind — Streamlit Frontend](#documind--streamlit-frontend)
  - [Table of contents](#table-of-contents)
  - [Overview](#overview)
  - [Architecture](#architecture)
  - [Pages](#pages)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Configuration](#configuration)
  - [Running the app](#running-the-app)
    - [First-time setup checklist](#first-time-setup-checklist)
  - [Project structure](#project-structure)
  - [Troubleshooting](#troubleshooting)
  - [Security notes](#security-notes)

---

## Overview

DocuMind is a retrieval-augmented generation (RAG) system covering three
document tiers — **Kubernetes documentation**, **internal policy
documents**, and **support tickets** — with hybrid (dense + sparse)
retrieval, cross-encoder reranking, parent-section expansion, scope and
confidence guardrails, and a RAGAS-style evaluation suite.

This frontend gives non-CLI users (analysts, support staff, stakeholders)
a way to interact with that backend: ask questions, inspect retrieval
quality, ingest new documents, run evaluations, and audit past
inferences — all through a browser.

## Architecture

```
question
  │
  ▼
generation.check_scope()          ── out of scope ──► refuse (no retrieval call)
  │ in scope
  ▼
retrieval.retrieve()              [hybrid dense+sparse search, RRF fusion,
  │                                 cross-encoder rerank, parent expansion]
  ▼
generation.evaluate_guardrails()  ── low confidence ──► refuse
  │ confident enough
  ▼
generation.generate()             [LCEL chain via OpenRouter]
  │
  ▼
generation.log_inference()        [SQLite inference_logs]
  │
  ▼
answer + inline citations  ──►  rendered in the Streamlit UI
```

The Streamlit app (`app.py`) sits entirely at the last step: it calls
`generation.answer_question()` (which runs the whole pipeline above) and
renders the result. Every other page maps to a specific backend
capability — see [Pages](#pages) below.

## Pages

| Page | Backend call(s) | Purpose |
|---|---|---|
| 🏠 **Dashboard** | `embeddings.get_client()`, `generation.fetch_recent()` | System health at a glance — Qdrant connectivity, indexed chunk count, recent activity. |
| 💬 **Ask DocuMind** | `generation.answer_question()` | Core Q&A experience — grounded answers, inline citations, refusal reasons, retrieved chunks. |
| 🔎 **Retrieval Explorer** | `retrieval.retrieve()` | Inspect hybrid retrieval and reranking in isolation, without an LLM call. |
| 🛡️ **Guardrails Inspector** | `generation.check_scope()` | Test scope classification for a question before it reaches retrieval. |
| 📥 **Document Ingestion** | `ingestion.run_docs/run_policy/run_support/run_ingestion()`, `embeddings.run_embedding()` | Upload new source files, re-chunk, and re-embed into Qdrant. |
| 📊 **Evaluation Dashboard** | `evaluation.run_evaluation()`, `save_report()`, `calibrate_threshold()` | Run the RAGAS-style eval suite and review scores, per-tier breakdown, and past reports. |
| 🧾 **Inference Logs** | `generation.fetch_recent()` | Audit trail of every inference: question, retrieval scores, prompt, answer, latency. |
| ⚙️ **System Settings** | `config.*`, `embeddings.get_client()` | View effective configuration and test Qdrant connectivity. |
| ℹ️ **About** | — | Static architecture explainer. |

## Prerequisites

- Python 3.10+
- A running [Qdrant](https://qdrant.tech/) instance (local or remote)
- An [OpenRouter](https://openrouter.ai/) API key (used for generation
  and the RAGAS judge model)
- The DocuMind backend modules (`config.py`, `embeddings.py`,
  `ingestion.py`, `retrieval.py`, `generation.py`, `evaluation.py`) in
  the same directory as `app.py`

## Installation

```bash
git clone <your-repo-url>
cd DocuMind

python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
pip install streamlit pandas    # if not already in requirements.txt
```

## Configuration

All configuration is read from `config.py`, which is env-var
overridable. Set these before starting the app:

```bash
# Required
export OPENROUTER_API_KEY=sk-or-v1-...

# Optional (defaults shown)
export DOCUMIND_QDRANT_URL=http://localhost:6333
export DOCUMIND_QDRANT_HYBRID_COLLECTION=documind_chunks_hybrid
export DOCUMIND_GENERATOR_MODEL=deepseek/deepseek-chat-v3
export DOCUMIND_JUDGE_MODEL=deepseek/deepseek-chat-v3
export DOCUMIND_REFUSAL_THRESHOLD=-9.64
export DOCUMIND_RERANK_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
export DOCUMIND_LOG_LEVEL=INFO
```

Add these to your shell profile (`~/.bashrc`, `~/.zshrc`) to avoid
re-exporting them every session.

Bring up Qdrant locally if you don't already have an instance:

```bash
docker run -p 6333:6333 qdrant/qdrant
```

## Running the app

```bash
streamlit run app.py
```

Streamlit will print a local URL (default `http://localhost:8501`) —
open it in your browser. To bind to all interfaces (e.g. running inside
a container or VM):

```bash
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

Check the sidebar for a **backend loaded** badge to confirm the backend
modules imported successfully. A **backend import failed** badge means
`app.py` isn't in the same directory as the backend modules, or a
dependency is missing — expand the error message on any page for
details.

### First-time setup checklist

1. ✅ Qdrant is running and reachable
2. ✅ `OPENROUTER_API_KEY` is set
3. ✅ Documents are ingested (**Document Ingestion** page, or the
   existing `python cli.py ingest && python cli.py embed`)
4. ✅ Ask a question on the **Ask DocuMind** page to confirm end-to-end
   behavior

## Project structure

```
DocuMind/
├── app.py                  # Streamlit frontend (this app)
├── config.py                # Settings — models, Qdrant, guardrails, paths
├── embeddings.py             # Dense + sparse embeddings, Qdrant client/vector store
├── ingestion.py               # Chunking pipeline for all three document tiers
├── retrieval.py                # Hybrid search, RRF fusion, reranking, parent expansion
├── generation.py                # Guardrails, LLM calls, inference logging, orchestration
├── evaluation.py                 # RAGAS-style evaluation suite, threshold calibration
├── data/
│   ├── docs/                      # Raw Kubernetes docs (markdown)
│   ├── filings/                   # Raw policy documents (markdown)
│   ├── support_qa/                # Raw support tickets (JSONL)
│   ├── processed/                 # Chunked output, parent sections, inference log DB
│   ├── gold_eval/                 # Gold QA set for evaluation
│   └── eval_results/               # Saved evaluation reports (JSON)
└── requirements.txt
```

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `RuntimeError: OPENROUTER_API_KEY is not set` | Env var not exported in this shell/session | `export OPENROUTER_API_KEY=...` before launching Streamlit |
| Sidebar shows **backend import failed** | `app.py` not in the same folder as the backend modules, or a missing dependency | Confirm working directory; `pip install -r requirements.txt` |
| Dashboard shows **Qdrant: Unreachable** | Qdrant isn't running, or `DOCUMIND_QDRANT_URL` is wrong | `docker run -p 6333:6333 qdrant/qdrant`; check the URL in **System Settings** |
| `gio: http://localhost:8501: Operation not supported` on startup | Streamlit trying (and failing) to auto-open a browser in a headless/WSL environment | Harmless — open the printed URL manually in your browser |
| Every question is refused as **out of scope** | No documents ingested yet, or scope keywords don't match your corpus | Run ingestion on the **Document Ingestion** page; check `SCOPE_KEYWORDS` in **System Settings** |
| Every question is refused as **low confidence** | Refusal threshold too strict for your corpus/model | Run **Calibrate refusal threshold** on the **Evaluation Dashboard** |

## Security notes

- `OPENROUTER_API_KEY` is read from the environment only — never
  hardcode it in `config.py` or commit it to version control.
- The **Document Ingestion** page writes uploaded files directly to disk
  and can trigger a full Qdrant collection rebuild — restrict access to
  this page in any multi-user deployment.
- Inference logs (including full prompts) are stored locally in SQLite
  at `INFERENCE_LOG_DB_PATH` — treat this file as containing
  potentially sensitive user queries.

  
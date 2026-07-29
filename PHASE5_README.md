# DocuMind — Phase 5: Generation, Guardrails, Evaluation

This delivers Phase 5 of the DocuMind build guide, sitting on top of the
already-implemented Phase 1-4 (data, ingestion/chunking, Qdrant hybrid
embeddings, hybrid retrieval + rerank).

**New in this delivery:** the `Generation/` package. See
`Generation/README.md` for the full design writeup (architecture diagram,
per-file responsibilities, design-decision rationale, environment
variables, and exact run instructions).

## Quick start

```bash
cd DocuMind
pip install -r Generation/requirements.txt --break-system-packages   # + Embedding/Retrieval reqs if not already installed

# 1. Start Ollama and pull the local generator model
ollama serve
ollama pull qwen3:8b

# 2. Make sure Qdrant is running with the Phase 4 hybrid collection populated

# 3. Calibrate the refusal threshold against the gold set (recommended first step)
python -m Generation.run_phase5 calibrate

# 4. Ask a question
python -m Generation.run_phase5 ask "What are the warnings about CPU manager for k8s 1.26+?"

# 5. Run the full evaluation (writes data/eval_results/ragas_eval_<ts>.json)
python -m Generation.run_phase5 eval
```

## What's included

```
Generation/
├── __init__.py
├── config.py           # models, thresholds, paths (all env-overridable)
├── prompts.py           # generation prompt + 4 RAGAS judge prompts
├── llm_client.py         # Ollama /api/chat wrapper (generate + judge)
├── guardrails.py         # scope check + refusal-on-low-confidence
├── logging_db.py         # SQLite inference_logs (feeds Phase 6)
├── generate.py           # orchestrator (Retrieval -> guardrails -> LLM -> log)
├── ragas_eval.py          # local RAGAS-style evaluator vs. gold set
├── run_phase5.py          # CLI: ask / eval / calibrate
├── requirements.txt
└── README.md              # full design writeup
```

## Verified in this sandbox

- All modules pass `python -m py_compile`.
- `logging_db.py` verified end-to-end against a real SQLite file (schema
  creation, insert, read-back).
- `prompts.py` template rendering verified against sample chunks.
- `guardrails.py` keyword-matching path verified with representative
  in-scope/out-of-scope questions.

**Not verifiable here** (sandbox has no route to a local Ollama server or
to huggingface.co, same constraint as the Phase 3/4 handoffs): the live
`llm_client.generate()`/`judge()` calls, and the BGE embedding fallback in
the scope check. Run the Quick Start above on your machine to confirm the
live path end-to-end before trusting it fully — in particular, run
`calibrate` before `eval`, since the default refusal threshold is a
starting guess, not a tuned value.

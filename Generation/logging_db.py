"""
Local SQLite logging for every inference: question, retrieved chunks,
retrieval/reranker scores, prompt, generated answer, latency. This is the
observability data Phase 6's dashboard/cache reads from -- keep the schema
stable, since Phase 6 depends on these exact column names.
"""
from __future__ import annotations
import json
import os
import sqlite3
import time
import uuid
from contextlib import closing
from dataclasses import dataclass, field

from . import config

log = config.get_generation_logger("logging_db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS inference_logs (
    id                  TEXT PRIMARY KEY,
    ts_unix             REAL NOT NULL,
    question            TEXT NOT NULL,
    guardrail_reason    TEXT,               -- NULL | 'out_of_scope' | 'low_confidence'
    retrieved_chunks    TEXT NOT NULL,       -- JSON list of {chunk_id, citation, fusion_score, rerank_score}
    top_rerank_score    REAL,
    prompt              TEXT,
    answer              TEXT NOT NULL,
    generator_model     TEXT,
    retrieval_latency_s REAL,
    rerank_latency_s    REAL,
    generation_latency_s REAL,
    total_latency_s     REAL
);
CREATE INDEX IF NOT EXISTS idx_inference_logs_ts ON inference_logs(ts_unix);
"""


@dataclass
class InferenceLogEntry:
    question: str
    answer: str
    retrieved_chunks: list[dict]
    guardrail_reason: str | None = None
    top_rerank_score: float | None = None
    prompt: str | None = None
    generator_model: str | None = None
    retrieval_latency_s: float | None = None
    rerank_latency_s: float | None = None
    generation_latency_s: float | None = None
    total_latency_s: float | None = None
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    ts_unix: float = field(default_factory=time.time)


def _connect() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(config.INFERENCE_LOG_DB_PATH) or ".", exist_ok=True)
    conn = sqlite3.connect(config.INFERENCE_LOG_DB_PATH)
    conn.executescript(SCHEMA)
    return conn


def log_inference(entry: InferenceLogEntry) -> None:
    with closing(_connect()) as conn:
        conn.execute(
            """INSERT INTO inference_logs (
                id, ts_unix, question, guardrail_reason, retrieved_chunks,
                top_rerank_score, prompt, answer, generator_model,
                retrieval_latency_s, rerank_latency_s, generation_latency_s,
                total_latency_s
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                entry.id,
                entry.ts_unix,
                entry.question,
                entry.guardrail_reason,
                json.dumps(entry.retrieved_chunks),
                entry.top_rerank_score,
                entry.prompt,
                entry.answer,
                entry.generator_model,
                entry.retrieval_latency_s,
                entry.rerank_latency_s,
                entry.generation_latency_s,
                entry.total_latency_s,
            ),
        )
        conn.commit()
    log.info("logged_inference id=%s guardrail=%s", entry.id, entry.guardrail_reason)


def fetch_recent(limit: int = 20) -> list[dict]:
    with closing(_connect()) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM inference_logs ORDER BY ts_unix DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]

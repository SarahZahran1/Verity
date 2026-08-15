
from __future__ import annotations

import glob
import json
import os
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import config
import evaluation
import generation
import retrieval

log = config.get_logger("api")

app = FastAPI(
    title="Verity API",
    description="Grounded enterprise RAG platform -- HTTP backend.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# Schemas


class ChatTurn(BaseModel):
    question: str
    answer: str


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)
    history: list[ChatTurn] = Field(default_factory=list)
    top_k: int = Field(default=config.FINAL_TOP_K, ge=1, le=20)
    log_to_db: bool = True


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int = Field(default=config.FINAL_TOP_K, ge=1, le=20)
    log_to_db: bool = True


class RetrieveRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int = Field(default=config.FINAL_TOP_K, ge=1, le=20)
    expand_parents: bool = False


class RetrievedChunkOut(BaseModel):
    chunk_id: str
    text: str
    rerank_score: float
    fusion_score: float
    doc_title: Optional[str] = None
    section_heading: Optional[str] = None
    source_path: Optional[str] = None
    tier: Optional[str] = None
    admonition_type: Optional[str] = None
    citation: str


class AnswerResponse(BaseModel):
    question: str
    answer: str
    citations: list[str] = Field(default_factory=list)
    refused: bool = False
    refusal_reason: Optional[str] = None
    retrieved: list[RetrievedChunkOut] = Field(default_factory=list)
    total_latency_s: float = 0.0
    intent: str = "new_question"
    used_rag: bool = True


# Helpers


def _serialize_answer(result: generation.Answer) -> AnswerResponse:
    return AnswerResponse(
        question=result.question,
        answer=result.answer,
        citations=result.citations,
        refused=result.refused,
        refusal_reason=result.refusal_reason,
        retrieved=[
            RetrievedChunkOut(
                chunk_id=r.chunk_id,
                text=r.text,
                rerank_score=r.rerank_score,
                fusion_score=r.fusion_score,
                doc_title=r.doc_title,
                section_heading=r.section_heading,
                source_path=r.source_path,
                tier=r.tier,
                admonition_type=r.admonition_type,
                citation=r.citation,
            )
            for r in result.retrieved
        ],
        total_latency_s=result.total_latency_s,
        intent=result.intent,
        used_rag=result.used_rag,
    )


# Routes


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/chat", response_model=AnswerResponse)
def chat(req: ChatRequest) -> AnswerResponse:
    """Conversational entrypoint. Mirrors what app.py's chat page calls:
    routes chit-chat / ack / off-topic / follow-up / new-question, and
    only runs full retrieval+generation for new_question."""
    try:
        history = [t.model_dump() for t in req.history]
        result = generation.handle_message(
            req.question,
            history=history,
            top_k=req.top_k,
            log_to_db=req.log_to_db,
        )
    except generation.LLMError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e
    return _serialize_answer(result)


@app.post("/ask", response_model=AnswerResponse)
def ask(req: AskRequest) -> AnswerResponse:
    """Single-shot RAG: always runs scope check -> retrieval -> rerank ->
    guardrail -> generation, no conversational routing. Equivalent to
    `python cli.py ask "..."`."""
    try:
        result = generation.answer_question(
            req.question, top_k=req.top_k, log_to_db=req.log_to_db
        )
    except generation.LLMError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e
    return _serialize_answer(result)


@app.post("/retrieve", response_model=list[RetrievedChunkOut])
def retrieve(req: RetrieveRequest) -> list[RetrievedChunkOut]:
    """Retrieval only (hybrid search + rerank), no LLM call. Useful for
    inspecting what the pipeline would ground an answer on, or debugging
    the refusal threshold."""
    results = retrieval.retrieve(
        req.question, top_k=req.top_k, expand_parents=req.expand_parents
    )
    return [
        RetrievedChunkOut(
            chunk_id=r.chunk_id,
            text=r.text,
            rerank_score=r.rerank_score,
            fusion_score=r.fusion_score,
            doc_title=r.doc_title,
            section_heading=r.section_heading,
            source_path=r.source_path,
            tier=r.tier,
            admonition_type=r.admonition_type,
            citation=r.citation,
        )
        for r in results
    ]


@app.get("/logs")
def logs(limit: int = 20) -> list[dict]:
    return generation.fetch_recent(limit=limit)


@app.get("/eval/latest")
def eval_latest() -> dict:
    """Returns the most recently saved evaluation report from
    data/eval_results/, same file app.py's home page reads."""
    pattern = os.path.join(config.EVAL_RESULTS_DIR, "ragas_eval_*.json")
    files = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
    if not files:
        raise HTTPException(status_code=404, detail="No evaluation report found yet.")
    with open(files[0], encoding="utf-8") as f:
        return json.load(f)


@app.post("/eval/run")
def eval_run() -> dict:
    """Triggers a full RAGAS-style evaluation run against the gold set.
    This is slow (one LLM call per metric per gold question) -- intended
    for internal/admin use, not a request a normal client should make
    synchronously in production."""
    report = evaluation.run_evaluation()
    path = evaluation.save_report(report)
    return {"report_path": path, "overall": report["overall"]}

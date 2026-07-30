"""
DocuMind — Enterprise RAG Platform · Streamlit Frontend
=========================================================

This file is a FRONTEND ONLY. It contains no retrieval, embedding,
generation, guardrail, or evaluation logic — every one of those already
exists in the backend (config.py, embeddings.py, ingestion.py,
generation.py, retrieval.py, evaluation.py) and is simply imported and
called here.

--------------------------------------------------------------------------
PAGE MAP  (why it exists · what it calls · what the user can do)
--------------------------------------------------------------------------
1. Dashboard
   Why:    A single landing screen an operator can glance at to know the
           system is alive before trusting it with real questions.
   Calls:  embeddings.get_client(), client.count()/get_collections(),
           generation.fetch_recent()
   Does:   See collection health, point counts, recent activity at a glance.

2. Ask DocuMind
   Why:    The core product experience — this is what 95% of end users
           touch. Everything else in the app supports this page.
   Calls:  generation.answer_question()
   Does:   Ask a question, get a grounded answer with inline citations,
           see refusal reasons, inspect the retrieved chunks / rerank
           scores backing the answer, and browse chat history.

3. Retrieval Explorer
   Why:    Retrieval quality is the single biggest lever on RAG quality.
           Engineers/analysts need to inspect *just* retrieval — dense +
           sparse fusion, cross-encoder reranking, parent expansion —
           without paying for an LLM generation call.
   Calls:  retrieval.retrieve()
   Does:   Run retrieval only, inspect fusion vs. rerank scores per chunk,
           see parent-section expansion, compare top_k / expand_parents.

4. Guardrails Inspector
   Why:    Guardrails (scope + refusal) are a compliance-relevant feature
           in an enterprise deployment; teams need to be able to test and
           trust them independently of the full answer pipeline.
   Calls:  generation.check_scope()
   Does:   Test whether a question is judged in-scope before spending a
           retrieval/generation call on it — useful for tuning prompts
           and demonstrating guardrail behavior to stakeholders.

5. Document Ingestion
   Why:    Enterprise RAG platforms need a controlled way to add new
           source material without a CLI. This wraps the existing
           ingestion + embedding pipeline behind upload widgets.
   Calls:  ingestion.run_docs() / run_policy() / run_support() /
           run_ingestion(), embeddings.run_embedding()
   Does:   Upload files into the right tier (Kubernetes docs / policy /
           support tickets), re-chunk, and re-embed into Qdrant.

6. Evaluation Dashboard
   Why:    "Is the system still good?" needs a real, repeatable answer —
           this is the RAGAS-style eval suite already built, surfaced so
           a non-CLI user can trigger and read it.
   Calls:  evaluation.run_evaluation(), evaluation.save_report(),
           evaluation.calibrate_threshold()
   Does:   Run an eval pass against the gold set, view faithfulness /
           relevance / precision / recall, per-tier breakdown, refusal
           accuracy on adversarial rows, and browse past eval reports.

7. Inference Logs
   Why:    Every production RAG system needs an audit trail: what was
           asked, what was retrieved, what was answered, how confident
           the system was, and how long it took.
   Calls:  generation.fetch_recent()
   Does:   Browse recent inferences, filter by guardrail outcome, drill
           into the retrieved chunks and prompt for any single call.

8. System Settings
   Why:    Operators need visibility into the *effective* configuration
           (models, thresholds, Qdrant target) without reading config.py,
           especially since most of it is env-var overridable.
   Calls:  config.* (read-only), embeddings.get_client()
   Does:   View current configuration, run a live Qdrant connectivity
           check, adjust frontend-only session preferences (top_k,
           whether to log inferences).

9. About
   Why:    Explains the architecture and guardrail design to new users /
           stakeholders reading over someone's shoulder.
   Calls:  none (static content mirrored from the backend README)
   Does:   Read a short architecture explainer.
--------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import time
import traceback
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import streamlit as st

# ---------------------------------------------------------------------------
# Backend imports — these are the ONLY places business logic comes from.
# Nothing in this file re-implements retrieval, generation, guardrails,
# ingestion, or evaluation.
# ---------------------------------------------------------------------------
try:
    import config
    import embeddings
    import generation
    import retrieval
    import ingestion
    import evaluation
    BACKEND_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - surfaced in the UI instead
    BACKEND_IMPORT_ERROR = exc


# ===========================================================================
# Page config + light theming
# ===========================================================================

st.set_page_config(
    page_title="DocuMind — Enterprise RAG Platform",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    .block-container { padding-top: 2rem; padding-bottom: 3rem; }
    .doc-badge {
        display: inline-block; padding: 2px 10px; border-radius: 12px;
        font-size: 0.75rem; font-weight: 600; margin-right: 6px;
    }
    .badge-ok      { background: #dcfce7; color: #166534; }
    .badge-warn    { background: #fef9c3; color: #854d0e; }
    .badge-err     { background: #fee2e2; color: #991b1b; }
    .badge-neutral { background: #e5e7eb; color: #374151; }
    .chunk-card {
        border: 1px solid #e5e7eb; border-radius: 10px; padding: 12px 16px;
        margin-bottom: 10px; background: #fafafa;
    }
    .citation-pill {
        display: inline-block; background: #eef2ff; color: #3730a3;
        border-radius: 6px; padding: 1px 8px; font-size: 0.78rem;
        margin-right: 4px; font-family: monospace;
    }
    .metric-caption { color: #6b7280; font-size: 0.8rem; }
    section[data-testid="stSidebar"] .stRadio label { font-size: 0.95rem; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ===========================================================================
# Session state
# ===========================================================================

def init_session_state() -> None:
    defaults = {
        "chat_history": [],       # list[dict]: question/answer turns for Ask page
        "top_k": 5,
        "log_to_db": True,
        "expand_parents": True,
        "last_retrieval": None,   # cache of last Retrieval Explorer run
        "eval_report": None,      # cache of last evaluation run this session
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


init_session_state()


# ===========================================================================
# Generic helpers
# ===========================================================================

@contextmanager
def safe_run(spinner_text: str, success_text: str | None = None):
    """Wrap a backend call with a spinner + consistent error handling.

    Usage:
        with safe_run("Retrieving...") as ok:
            if ok:
                result = retrieval.retrieve(...)
    """
    placeholder = st.empty()
    with placeholder, st.spinner(spinner_text):
        try:
            yield True
            if success_text:
                st.toast(success_text, icon="✅")
        except Exception as exc:  # noqa: BLE001 - user-facing error boundary
            st.error(f"**Something went wrong:** {exc}")
            with st.expander("Technical details"):
                st.code(traceback.format_exc())
            yield False


def score_badge(score: float | None, threshold: float) -> str:
    if score is None:
        return '<span class="doc-badge badge-neutral">no score</span>'
    cls = "badge-ok" if score >= threshold else "badge-err"
    return f'<span class="doc-badge {cls}">rerank {score:.2f}</span>'


def guardrail_badge(reason: str | None) -> str:
    if reason is None:
        return '<span class="doc-badge badge-ok">answered</span>'
    if reason == "out_of_scope":
        return '<span class="doc-badge badge-warn">out of scope</span>'
    if reason == "low_confidence":
        return '<span class="doc-badge badge-err">low confidence</span>'
    return f'<span class="doc-badge badge-neutral">{reason}</span>'


def fmt_ts(ts: float | None) -> str:
    if not ts:
        return "—"
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def render_retrieved_chunks(retrieved: list, threshold: float) -> None:
    """Render RetrievalResult objects as inspectable cards with scores."""
    if not retrieved:
        st.info("No chunks were retrieved.")
        return
    for i, r in enumerate(retrieved, start=1):
        rerank_score = getattr(r, "rerank_score", None)
        fusion_score = getattr(r, "fusion_score", None)
        citation = getattr(r, "citation", "unknown source")
        chunk_id = getattr(r, "chunk_id", f"chunk_{i}")
        text = getattr(r, "text", "")
        parent_id = getattr(r, "parent_id", None)

        st.markdown(
            f"""<div class="chunk-card">
            <span class="citation-pill">{chunk_id}</span>
            {score_badge(rerank_score, threshold)}
            <span class="doc-badge badge-neutral">fusion {fusion_score:.3f}</span>
            <div class="metric-caption" style="margin-top:6px;">source: {citation}
            {" · parent: " + str(parent_id) if parent_id else ""}</div>
            </div>""".replace(
                f'<span class="doc-badge badge-neutral">fusion {fusion_score:.3f}</span>',
                (f'<span class="doc-badge badge-neutral">fusion {fusion_score:.3f}</span>'
                 if fusion_score is not None else ""),
            ),
            unsafe_allow_html=True,
        )
        with st.expander(f"View chunk text ({len(text)} chars)"):
            st.write(text)


def backend_ready() -> bool:
    if BACKEND_IMPORT_ERROR is not None:
        st.error(
            "The backend modules could not be imported, so the UI cannot "
            "call any real functionality yet."
        )
        with st.expander("Import error details"):
            st.code(str(BACKEND_IMPORT_ERROR))
        st.caption(
            "Make sure this app runs from the backend project root (same "
            "folder as config.py / embeddings.py / generation.py / "
            "retrieval.py / ingestion.py / evaluation.py) and that "
            "dependencies are installed."
        )
        return False
    return True


# ===========================================================================
# Page: Dashboard
# ===========================================================================

def page_dashboard() -> None:
    st.title("🧠 DocuMind Dashboard")
    st.caption("Enterprise RAG platform — Kubernetes docs · policy · support tickets")

    if not backend_ready():
        return

    col1, col2, col3, col4 = st.columns(4)

    qdrant_ok, point_count, collections = None, None, []
    with st.spinner("Checking Qdrant connection..."):
        try:
            client = embeddings.get_client()
            collections = [c.name for c in client.get_collections().collections]
            if config.QDRANT_COLLECTION in collections:
                point_count = client.count(
                    collection_name=config.QDRANT_COLLECTION, exact=True
                ).count
            qdrant_ok = True
        except Exception as exc:
            qdrant_ok = False
            qdrant_error = str(exc)

    with col1:
        st.metric("Qdrant status", "Connected" if qdrant_ok else "Unreachable")
    with col2:
        st.metric(
            "Indexed chunks",
            point_count if point_count is not None else "—",
        )
    with col3:
        st.metric("Generator model", config.GENERATOR_MODEL)
    with col4:
        st.metric("Refusal threshold", f"{config.REFUSAL_RERANK_THRESHOLD:.2f}")

    if not qdrant_ok:
        st.warning(
            f"Could not reach Qdrant at `{config.QDRANT_URL}`: {qdrant_error}\n\n"
            "Check `DOCUMIND_QDRANT_URL` / connectivity, or use **System "
            "Settings** to re-test."
        )

    st.divider()

    left, right = st.columns([3, 2])

    with left:
        st.subheader("Recent activity")
        with safe_run("Loading recent inference logs...") as ok:
            if ok:
                rows = generation.fetch_recent(limit=8)
                if not rows:
                    st.info("No inferences logged yet — ask a question to get started.")
                else:
                    for row in rows:
                        st.markdown(
                            f"{guardrail_badge(row.get('guardrail_reason'))} "
                            f"**{row['question']}**  \n"
                            f"<span class='metric-caption'>{fmt_ts(row.get('ts_unix'))} · "
                            f"{row.get('total_latency_s') or 0:.2f}s</span>",
                            unsafe_allow_html=True,
                        )
                        st.divider()

    with right:
        st.subheader("Supported domains")
        for domain, keywords in config.SCOPE_KEYWORDS.items():
            with st.container(border=True):
                st.markdown(f"**{domain.title()}**")
                st.caption(", ".join(keywords[:8]) + ("…" if len(keywords) > 8 else ""))

        st.subheader("Collections in Qdrant")
        if collections:
            for c in collections:
                marker = "🟢" if c == config.QDRANT_COLLECTION else "⚪"
                st.markdown(f"{marker} `{c}`")
        else:
            st.caption("No collections found (or Qdrant unreachable).")


# ===========================================================================
# Page: Ask DocuMind
# ===========================================================================

def page_ask() -> None:
    st.title("💬 Ask DocuMind")
    st.caption(
        "Ask a question grounded in the ingested Kubernetes docs, policy "
        "documents, and support tickets. Answers are refused when the "
        "question is out of scope or retrieval confidence is too low."
    )

    if not backend_ready():
        return

    with st.form("ask_form", clear_on_submit=False):
        question = st.text_area(
            "Your question",
            placeholder="e.g. What are the warnings about CPU manager for k8s 1.26+?",
            height=90,
        )
        c1, c2, c3 = st.columns([1, 1, 2])
        with c1:
            top_k = st.number_input(
                "top_k", min_value=1, max_value=20, value=st.session_state.top_k
            )
        with c2:
            log_to_db = st.checkbox("Log this inference", value=st.session_state.log_to_db)
        with c3:
            submitted = st.form_submit_button("Ask", type="primary", use_container_width=True)

    if submitted:
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            st.session_state.top_k = int(top_k)
            st.session_state.log_to_db = log_to_db
            with safe_run("Checking scope, retrieving context, and generating an answer...") as ok:
                if ok:
                    t0 = time.time()
                    result = generation.answer_question(
                        question, top_k=int(top_k), log_to_db=log_to_db
                    )
                    st.session_state.chat_history.insert(
                        0,
                        {
                            "question": question,
                            "result": result,
                            "wall_time": time.time() - t0,
                        },
                    )

    if not st.session_state.chat_history:
        st.info("Ask a question above to see a grounded answer with citations.")
        return

    st.divider()
    st.subheader("Conversation")

    for turn in st.session_state.chat_history:
        result = turn["result"]
        with st.container(border=True):
            st.markdown(f"**Q: {turn['question']}**")
            st.markdown(guardrail_badge(result.refusal_reason), unsafe_allow_html=True)
            st.write(result.answer)

            if result.citations:
                st.markdown(
                    " ".join(f'<span class="citation-pill">{c}</span>' for c in result.citations),
                    unsafe_allow_html=True,
                )

            meta_cols = st.columns(4)
            meta_cols[0].markdown(
                f"<span class='metric-caption'>Total latency: "
                f"{result.total_latency_s:.2f}s</span>", unsafe_allow_html=True
            )
            meta_cols[1].markdown(
                f"<span class='metric-caption'>Chunks retrieved: "
                f"{len(result.retrieved)}</span>", unsafe_allow_html=True
            )
            top_score = result.retrieved[0].rerank_score if result.retrieved else None
            meta_cols[2].markdown(
                f"<span class='metric-caption'>Top rerank score: "
                f"{top_score:.2f}</span>" if top_score is not None
                else "<span class='metric-caption'>Top rerank score: —</span>",
                unsafe_allow_html=True,
            )
            meta_cols[3].markdown(
                f"<span class='metric-caption'>Refused: {result.refused}</span>",
                unsafe_allow_html=True,
            )

            if result.retrieved:
                with st.expander("View retrieved chunks & scores"):
                    render_retrieved_chunks(
                        result.retrieved, config.REFUSAL_RERANK_THRESHOLD
                    )

    if st.button("Clear conversation"):
        st.session_state.chat_history = []
        st.rerun()


# ===========================================================================
# Page: Retrieval Explorer
# ===========================================================================

def page_retrieval_explorer() -> None:
    st.title("🔎 Retrieval Explorer")
    st.caption(
        "Run hybrid (dense + sparse) retrieval with cross-encoder reranking "
        "directly — no generation call, useful for debugging retrieval "
        "quality in isolation."
    )

    if not backend_ready():
        return

    with st.form("retrieval_form"):
        query = st.text_input("Query", placeholder="CPU manager warnings")
        c1, c2 = st.columns(2)
        with c1:
            top_k = st.slider("top_k", min_value=1, max_value=20, value=st.session_state.top_k)
        with c2:
            expand_parents = st.checkbox(
                "Expand to parent sections", value=st.session_state.expand_parents
            )
        run = st.form_submit_button("Retrieve", type="primary")

    if run:
        if not query.strip():
            st.warning("Please enter a query.")
        else:
            st.session_state.top_k = top_k
            st.session_state.expand_parents = expand_parents
            with safe_run("Running hybrid retrieval + reranking...") as ok:
                if ok:
                    t0 = time.time()
                    results = retrieval.retrieve(
                        query, top_k=top_k, expand_parents=expand_parents
                    )
                    st.session_state.last_retrieval = {
                        "query": query,
                        "results": results,
                        "latency": time.time() - t0,
                    }

    cache = st.session_state.last_retrieval
    if not cache:
        st.info("Run a query to inspect fusion and rerank scores per chunk.")
        return

    st.divider()
    st.subheader(f"Results for: *{cache['query']}*")
    st.caption(f"Retrieved {len(cache['results'])} chunks in {cache['latency']:.2f}s")

    if cache["results"]:
        try:
            import pandas as pd

            df = pd.DataFrame(
                [
                    {
                        "chunk_id": r.chunk_id,
                        "citation": r.citation,
                        "fusion_score": round(r.fusion_score, 4)
                        if r.fusion_score is not None else None,
                        "rerank_score": round(r.rerank_score, 4)
                        if r.rerank_score is not None else None,
                        "parent_id": r.parent_id,
                    }
                    for r in cache["results"]
                ]
            )
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.bar_chart(df.set_index("chunk_id")["rerank_score"])
        except Exception:
            pass  # table/chart are a bonus, cards below always work

    render_retrieved_chunks(cache["results"], config.REFUSAL_RERANK_THRESHOLD)


# ===========================================================================
# Page: Guardrails Inspector
# ===========================================================================

def page_guardrails() -> None:
    st.title("🛡️ Guardrails Inspector")
    st.caption(
        "Scope checking runs before retrieval (cheap keyword match, with an "
        "embedding-centroid fallback). Test how a question is classified "
        "without spending a generation call."
    )

    if not backend_ready():
        return

    question = st.text_input(
        "Question to test", placeholder="What's the weather like today?"
    )
    if st.button("Check scope", type="primary"):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            with safe_run("Evaluating scope guardrail...") as ok:
                if ok:
                    in_scope = generation.check_scope(question)
                    if in_scope:
                        st.success("✅ In scope — this question would proceed to retrieval.")
                    else:
                        st.error(
                            "🚫 Out of scope — this question would be refused "
                            f"immediately with: \"{config.OUT_OF_SCOPE_MESSAGE}\""
                        )

    st.divider()
    st.subheader("Guardrail configuration (read-only)")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Scope check**")
        st.json({
            "embedding_threshold": config.SCOPE_EMBEDDING_THRESHOLD,
            "domains": list(config.SCOPE_KEYWORDS.keys()),
        })
    with c2:
        st.markdown("**Refusal-on-low-confidence**")
        st.json({
            "rerank_threshold": config.REFUSAL_RERANK_THRESHOLD,
            "refusal_message": config.REFUSAL_MESSAGE,
        })
    st.caption(
        "To change these thresholds, set the corresponding environment "
        "variables and restart the app (see System Settings), or run "
        "calibration from the Evaluation Dashboard."
    )


# ===========================================================================
# Page: Document Ingestion
# ===========================================================================

def page_ingestion() -> None:
    st.title("📥 Document Ingestion")
    st.caption(
        "Add new source material to the knowledge base. Files are saved "
        "into the tier's raw data folder, then chunked and embedded using "
        "the existing ingestion + embedding pipeline."
    )

    if not backend_ready():
        return

    tier = st.selectbox(
        "Tier",
        ["Kubernetes docs (markdown)", "Policy documents (markdown)", "Support tickets (JSONL)"],
    )

    if tier == "Kubernetes docs (markdown)":
        target_dir = ingestion.RAW / "docs"
        uploads = st.file_uploader(
            "Upload .md files", type=["md"], accept_multiple_files=True
        )
        run_label = "Re-chunk Kubernetes docs tier"
        runner = lambda: ingestion.run_docs(
            str(target_dir), str(ingestion.PROCESSED / "chunks_docs.jsonl")
        )
    elif tier == "Policy documents (markdown)":
        target_dir = ingestion.RAW / "filings"
        uploads = st.file_uploader(
            "Upload .md files", type=["md"], accept_multiple_files=True
        )
        run_label = "Re-chunk policy tier"
        runner = lambda: ingestion.run_policy(
            str(target_dir), str(ingestion.PROCESSED / "chunks_policy.jsonl")
        )
    else:
        target_dir = ingestion.RAW / "support_qa"
        uploads = st.file_uploader(
            "Upload a support_tickets.jsonl file", type=["jsonl"], accept_multiple_files=False
        )
        run_label = "Re-chunk support tier"
        runner = lambda: ingestion.run_support(
            str(target_dir / "support_tickets.jsonl"),
            str(ingestion.PROCESSED / "chunks_support.jsonl"),
        )

    save_col, run_col = st.columns(2)

    with save_col:
        if st.button("Save uploaded file(s) to raw data folder", type="secondary"):
            files = uploads if isinstance(uploads, list) else ([uploads] if uploads else [])
            if not files:
                st.warning("Upload at least one file first.")
            else:
                with safe_run("Saving uploaded files...") as ok:
                    if ok:
                        target_dir.mkdir(parents=True, exist_ok=True)
                        for f in files:
                            dest = target_dir / (
                                f.name if tier != "Support tickets (JSONL)"
                                else "support_tickets.jsonl"
                            )
                            dest.write_bytes(f.getvalue())
                        st.success(f"Saved {len(files)} file(s) to `{target_dir}`.")

    with run_col:
        if st.button(run_label, type="primary"):
            with safe_run(f"Running: {run_label}...") as ok:
                if ok:
                    chunks = runner()
                    st.success(f"Produced {len(chunks)} chunks for this tier.")

    st.divider()
    st.subheader("Full pipeline")
    st.caption(
        "Re-runs ingestion for **all three tiers**, merges outputs into "
        "`chunks_all.jsonl` / `parents_all.jsonl`, and validates against "
        "the gold set if present."
    )
    if st.button("Run full ingestion (all tiers)"):
        with safe_run("Running full ingestion across all tiers...") as ok:
            if ok:
                all_chunks = ingestion.run_ingestion()
                st.success(f"Ingestion complete — {len(all_chunks)} total chunks.")

    st.divider()
    st.subheader("Embed & index into Qdrant")
    st.caption(
        "Embeds every chunk in `chunks_all.jsonl` (dense + sparse) and "
        "upserts into the hybrid Qdrant collection. This **rebuilds the "
        "collection from scratch** — run it after ingestion changes."
    )
    confirm = st.checkbox("I understand this rebuilds the Qdrant collection")
    if st.button("Run embedding pipeline", disabled=not confirm):
        with safe_run("Embedding chunks and upserting into Qdrant (this may take a while)...") as ok:
            if ok:
                count = embeddings.run_embedding()
                st.success(f"Qdrant collection now has {count} points.")


# ===========================================================================
# Page: Evaluation Dashboard
# ===========================================================================

def page_evaluation() -> None:
    st.title("📊 Evaluation Dashboard")
    st.caption(
        "RAGAS-style evaluation against the gold QA set: faithfulness, "
        "answer relevance, context precision, context recall, plus refusal "
        "accuracy on adversarial rows."
    )

    if not backend_ready():
        return

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Run full evaluation", type="primary"):
            with safe_run("Running evaluation against the gold set (this can take a while)...") as ok:
                if ok:
                    report = evaluation.run_evaluation()
                    path = evaluation.save_report(report)
                    st.session_state.eval_report = report
                    st.success(f"Evaluation complete. Report saved to `{path}`.")
    with c2:
        if st.button("Calibrate refusal threshold"):
            with safe_run("Sweeping refusal threshold against the gold set...") as ok:
                if ok:
                    suggested = evaluation.calibrate_threshold()
                    st.success(
                        f"Suggested `DOCUMIND_REFUSAL_THRESHOLD` ≈ **{suggested}** "
                        "(current: "
                        f"{config.REFUSAL_RERANK_THRESHOLD}). Set the env var and "
                        "restart to apply."
                    )

    report = st.session_state.eval_report
    if report:
        st.divider()
        st.subheader("Latest run (this session)")
        st.caption(
            f"{report['n_samples']} samples · generator: {report['generator_model']} · "
            f"judge: {report['judge_model']} · {fmt_ts(report['run_ts'])}"
        )

        overall = report["overall"]
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Faithfulness", f"{overall['faithfulness']:.2f}" if overall["faithfulness"] is not None else "—")
        m2.metric("Answer relevance", f"{overall['answer_relevance']:.2f}" if overall["answer_relevance"] is not None else "—")
        m3.metric("Context precision", f"{overall['context_precision']:.2f}" if overall["context_precision"] is not None else "—")
        m4.metric("Context recall", f"{overall['context_recall']:.2f}" if overall["context_recall"] is not None else "—")
        refusal_acc = report.get("refusal_accuracy_on_adversarial_set")
        m5.metric("Refusal accuracy (adversarial)", f"{refusal_acc:.2f}" if refusal_acc is not None else "—")

        st.subheader("By tier")
        try:
            import pandas as pd
            tier_df = pd.DataFrame(report["by_tier"]).T
            st.dataframe(tier_df, use_container_width=True)
        except Exception:
            st.json(report["by_tier"])

        with st.expander("View all individual samples"):
            st.dataframe(report["samples"], use_container_width=True)

    st.divider()
    st.subheader("Past evaluation reports")
    eval_dir = Path(config.EVAL_RESULTS_DIR)
    if eval_dir.exists():
        report_files = sorted(eval_dir.glob("ragas_eval_*.json"), reverse=True)
        if not report_files:
            st.caption("No saved reports yet.")
        for rf in report_files[:10]:
            with st.expander(rf.name):
                try:
                    data = json.loads(rf.read_text(encoding="utf-8"))
                    st.json(data.get("overall", {}))
                    st.caption(
                        f"Refusal accuracy (adversarial): "
                        f"{data.get('refusal_accuracy_on_adversarial_set')}"
                    )
                except Exception as e:
                    st.caption(f"Could not read report: {e}")
    else:
        st.caption(f"No eval results directory found at `{config.EVAL_RESULTS_DIR}` yet.")


# ===========================================================================
# Page: Inference Logs
# ===========================================================================

def page_logs() -> None:
    st.title("🧾 Inference Logs")
    st.caption(
        "Audit trail of every inference: question, guardrail outcome, "
        "retrieved chunks, rerank score, prompt, answer, and latency "
        "breakdown — read from the SQLite inference log."
    )

    if not backend_ready():
        return

    limit = st.slider("Number of recent logs", min_value=5, max_value=200, value=25)
    filter_option = st.selectbox(
        "Filter by outcome", ["All", "Answered", "Out of scope", "Low confidence"]
    )

    with safe_run("Loading inference logs...") as ok:
        if ok:
            rows = generation.fetch_recent(limit=limit)

            if filter_option == "Answered":
                rows = [r for r in rows if r.get("guardrail_reason") is None]
            elif filter_option == "Out of scope":
                rows = [r for r in rows if r.get("guardrail_reason") == "out_of_scope"]
            elif filter_option == "Low confidence":
                rows = [r for r in rows if r.get("guardrail_reason") == "low_confidence"]

            if not rows:
                st.info("No logs match this filter yet.")
            else:
                for row in rows:
                    with st.container(border=True):
                        header_cols = st.columns([5, 1])
                        header_cols[0].markdown(
                            f"{guardrail_badge(row.get('guardrail_reason'))} "
                            f"**{row['question']}**",
                            unsafe_allow_html=True,
                        )
                        header_cols[1].caption(fmt_ts(row.get("ts_unix")))

                        m = st.columns(4)
                        m[0].caption(f"Top rerank: {row.get('top_rerank_score')}")
                        m[1].caption(f"Retrieval: {row.get('retrieval_latency_s')}")
                        m[2].caption(f"Generation: {row.get('generation_latency_s')}")
                        m[3].caption(f"Total: {row.get('total_latency_s')}")

                        with st.expander("Answer & retrieved chunks"):
                            st.write(row.get("answer"))
                            try:
                                chunks = json.loads(row.get("retrieved_chunks") or "[]")
                                if chunks:
                                    st.dataframe(chunks, use_container_width=True, hide_index=True)
                            except Exception:
                                pass
                        with st.expander("Full prompt sent to the LLM"):
                            st.code(row.get("prompt") or "(no prompt — request was refused)")


# ===========================================================================
# Page: System Settings
# ===========================================================================

def page_settings() -> None:
    st.title("⚙️ System Settings")
    st.caption(
        "Effective configuration is read live from `config.py` (env-var "
        "overridable). This page is read-only for backend settings; "
        "session preferences below only affect this browser session."
    )

    if not backend_ready():
        return

    st.subheader("Live connectivity check")
    if st.button("Test Qdrant connection"):
        with safe_run("Contacting Qdrant...") as ok:
            if ok:
                client = embeddings.get_client()
                cols = client.get_collections().collections
                st.success(f"Connected. {len(cols)} collection(s) found.")
                st.json([c.name for c in cols])

    st.divider()
    st.subheader("Effective configuration")
    tabs = st.tabs(["Retrieval", "Generation", "Guardrails", "Qdrant", "Paths"])
    with tabs[0]:
        st.json({
            "RRF_K": config.RRF_K,
            "PREFETCH_LIMIT": config.PREFETCH_LIMIT,
            "FUSED_TOP_N": config.FUSED_TOP_N,
            "FINAL_TOP_K": config.FINAL_TOP_K,
            "RERANK_MODEL": config.RERANK_MODEL,
        })
    with tabs[1]:
        st.json({
            "GENERATOR_MODEL": config.GENERATOR_MODEL,
            "JUDGE_MODEL": config.JUDGE_MODEL,
            "GENERATION_TEMPERATURE": config.GENERATION_TEMPERATURE,
            "GENERATION_MAX_TOKENS": config.GENERATION_MAX_TOKENS,
            "GENERATION_NUM_CTX": config.GENERATION_NUM_CTX,
        })
    with tabs[2]:
        st.json({
            "REFUSAL_RERANK_THRESHOLD": config.REFUSAL_RERANK_THRESHOLD,
            "SCOPE_EMBEDDING_THRESHOLD": config.SCOPE_EMBEDDING_THRESHOLD,
            "REFUSAL_MESSAGE": config.REFUSAL_MESSAGE,
            "OUT_OF_SCOPE_MESSAGE": config.OUT_OF_SCOPE_MESSAGE,
        })
    with tabs[3]:
        st.json({
            "QDRANT_URL": config.QDRANT_URL,
            "QDRANT_COLLECTION": config.QDRANT_COLLECTION,
            "DENSE_VECTOR_NAME": config.DENSE_VECTOR_NAME,
            "SPARSE_VECTOR_NAME": config.SPARSE_VECTOR_NAME,
            "SPARSE_MODEL": config.SPARSE_MODEL,
            "EMBEDDING_MODEL": config.EMBEDDING_MODEL,
        })
    with tabs[4]:
        st.json({
            "CHUNKS_PATH": config.CHUNKS_PATH,
            "PARENTS_PATH": config.PARENTS_PATH,
            "INFERENCE_LOG_DB_PATH": config.INFERENCE_LOG_DB_PATH,
            "GOLD_EVAL_PATH": config.GOLD_EVAL_PATH,
            "EVAL_RESULTS_DIR": config.EVAL_RESULTS_DIR,
        })

    st.divider()
    st.subheader("Session preferences")
    st.session_state.top_k = st.number_input(
        "Default top_k for Ask / Retrieval Explorer",
        min_value=1, max_value=20, value=st.session_state.top_k,
    )
    st.session_state.log_to_db = st.checkbox(
        "Log inferences to the audit trail by default", value=st.session_state.log_to_db
    )
    st.caption("These only change defaults pre-filled on other pages.")


# ===========================================================================
# Page: About
# ===========================================================================

def page_about() -> None:
    st.title("ℹ️ About DocuMind")
    st.markdown(
        """
DocuMind is a retrieval-augmented generation platform covering three
document tiers: **Kubernetes documentation**, **internal policy
documents**, and **support tickets**.

**Pipeline**

1. A question first passes a **scope guardrail** — a cheap keyword match
   with an embedding-centroid fallback — before any retrieval work
   happens.
2. In-scope questions go through **hybrid retrieval**: dense (BGE) +
   sparse (BM25) search fused with Reciprocal Rank Fusion, then
   **cross-encoder reranking**, then **parent-section expansion** so the
   LLM sees full context, not just an isolated chunk.
3. A **confidence guardrail** refuses to answer if the top reranked chunk
   scores below a calibrated threshold, rather than letting the LLM
   hallucinate from weak context.
4. The LLM generates an answer **grounded only in retrieved context**,
   with mandatory inline citations back to source chunk IDs.
5. Every inference — answered or refused — is logged to a local SQLite
   audit trail with full latency breakdown.
6. A **RAGAS-style evaluation suite** scores the system against a gold
   question set on faithfulness, answer relevance, context precision, and
   context recall, plus refusal accuracy on adversarial questions.

**This application is the frontend only.** All retrieval, generation,
guardrail, ingestion, and evaluation logic lives in the backend modules
(`config.py`, `embeddings.py`, `ingestion.py`, `retrieval.py`,
`generation.py`, `evaluation.py`) and is called here, never reimplemented.
        """
    )
    st.divider()
    st.subheader("Page reference")
    st.table(
        {
            "Page": [
                "Dashboard", "Ask DocuMind", "Retrieval Explorer",
                "Guardrails Inspector", "Document Ingestion",
                "Evaluation Dashboard", "Inference Logs", "System Settings",
            ],
            "Backend function(s)": [
                "embeddings.get_client(), generation.fetch_recent()",
                "generation.answer_question()",
                "retrieval.retrieve()",
                "generation.check_scope()",
                "ingestion.run_docs/run_policy/run_support/run_ingestion(), embeddings.run_embedding()",
                "evaluation.run_evaluation(), save_report(), calibrate_threshold()",
                "generation.fetch_recent()",
                "config.*, embeddings.get_client()",
            ],
        }
    )


# ===========================================================================
# Sidebar navigation
# ===========================================================================

PAGES = {
    "🏠 Dashboard": page_dashboard,
    "💬 Ask DocuMind": page_ask,
    "🔎 Retrieval Explorer": page_retrieval_explorer,
    "🛡️ Guardrails Inspector": page_guardrails,
    "📥 Document Ingestion": page_ingestion,
    "📊 Evaluation Dashboard": page_evaluation,
    "🧾 Inference Logs": page_logs,
    "⚙️ System Settings": page_settings,
    "ℹ️ About": page_about,
}


def main() -> None:
    with st.sidebar:
        st.markdown("## 🧠 DocuMind")
        st.caption("Enterprise RAG Platform")
        st.divider()
        choice = st.radio("Navigate", list(PAGES.keys()), label_visibility="collapsed")
        st.divider()
        if BACKEND_IMPORT_ERROR is None:
            st.markdown(
                '<span class="doc-badge badge-ok">backend loaded</span>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<span class="doc-badge badge-err">backend import failed</span>',
                unsafe_allow_html=True,
            )
        st.caption(f"Model: {getattr(config, 'GENERATOR_MODEL', '—')}")

    PAGES[choice]()


if __name__ == "__main__":
    main()
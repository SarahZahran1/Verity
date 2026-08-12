"""
DocuMind — Enterprise RAG Platform · Streamlit Frontend (simplified)
=====================================================================
Two pages only:
  1. Home   — project name, short description, eval score, Start Chat button
  2. Chat   — ask anything, grounded answer with citations
"""

from __future__ import annotations

import json
import time
import traceback
from contextlib import contextmanager
from pathlib import Path

import streamlit as st

try:
    import config
    import generation
    BACKEND_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - surfaced in the UI instead
    BACKEND_IMPORT_ERROR = exc


st.set_page_config(
    page_title="DocuMind",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

CUSTOM_CSS = """
<style>
    .block-container { padding-top: 2rem; padding-bottom: 3rem; }
    .citation-pill {
        display: inline-block; background: #eef2ff; color: #3730a3;
        border-radius: 6px; padding: 1px 8px; font-size: 0.78rem;
        margin-right: 4px; font-family: monospace;
    }
    .metric-caption { color: #6b7280; font-size: 0.8rem; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def init_session_state() -> None:
    defaults = {
        "chat_history": [],
        "page": "home",
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


init_session_state()


@contextmanager
def safe_run(spinner_text: str, success_text: str | None = None):
    placeholder = st.empty()
    try:
        with placeholder, st.spinner(spinner_text):
            yield True
        if success_text:
            st.toast(success_text, icon="✅")
    except Exception as exc:
        st.error(f"**Something went wrong:** {exc}")
        with st.expander("Technical details"):
            st.code(traceback.format_exc())


def backend_ready() -> bool:
    if BACKEND_IMPORT_ERROR is not None:
        st.error("The backend modules could not be imported, so the app cannot run yet.")
        with st.expander("Import error details"):
            st.code(str(BACKEND_IMPORT_ERROR))
        return False
    return True


def get_latest_eval_score():
    """Return (score, source_label) for the latest available eval report, or (None, None)."""
    if not backend_ready():
        return None, None
    eval_dir = Path(config.EVAL_RESULTS_DIR)
    if not eval_dir.exists():
        return None, None
    report_files = sorted(eval_dir.glob("ragas_eval_*.json"), reverse=True)
    if not report_files:
        return None, None
    try:
        data = json.loads(report_files[0].read_text(encoding="utf-8"))
        score = data.get("overall", {}).get("faithfulness")
        return score, report_files[0].name
    except Exception:
        return None, None


# ===========================================================================
# Page: Home
# ===========================================================================

def page_home() -> None:
    st.markdown(
        "<h1 style='text-align:center; margin-top: 3rem;'>🧠 DocuMind</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='text-align:center; font-size: 1.1rem; color:#6b7280;'>"
        "Ask anything and get a grounded, easy-to-read answer — powered by "
        "retrieval over your documents."
        "</p>",
        unsafe_allow_html=True,
    )

    st.write("")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        with st.container(border=True):
            st.markdown("**📊 Eval set score**")
            score, source = get_latest_eval_score()
            if score is not None:
                st.metric("Faithfulness", f"{score:.2f}")
                st.caption(f"Source: {source}")
            else:
                st.info("No evaluation report found yet.")

        st.write("")
        if st.button("💬 Start Chat", type="primary", use_container_width=True):
            st.session_state.page = "chat"
            st.rerun()


# ===========================================================================
# Page: Chat
# ===========================================================================

def render_ref_expander(result) -> None:
    """Show only the paragraphs actually used to generate the answer."""
    if not (getattr(result, "citations", None) and getattr(result, "retrieved", None)):
        return
    cited_ids = set(result.citations)
    cited_chunks = [r for r in result.retrieved if r.chunk_id in cited_ids]
    if not cited_chunks:
        return
    with st.expander(f"ref ({len(cited_chunks)})"):
        for r in cited_chunks:
            st.markdown(f"**{getattr(r, 'citation', 'unknown source')}** · `{r.chunk_id}`")
            st.write(getattr(r, "parent_text", None) or getattr(r, "text", ""))
            st.divider()


def page_chat() -> None:
    top = st.columns([1, 6])
    with top[0]:
        if st.button("← Home"):
            st.session_state.page = "home"
            st.rerun()
    with top[1]:
        st.title("💬 Ask DocuMind")

    if not backend_ready():
        return

    # Render existing conversation, oldest first, in real chat bubbles.
    for turn in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(turn["question"])
        with st.chat_message("assistant"):
            if turn.get("error"):
                st.error(f"**Something went wrong:** {turn['error']}")
                with st.expander("Technical details"):
                    st.code(turn.get("traceback", ""))
            else:
                st.write(turn["result"].answer)
                render_ref_expander(turn["result"])

    # Chat input pinned at the bottom -- the natural chat UX.
    question = st.chat_input("Ask anything...")

    if question and question.strip():
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    t0 = time.time()
                    # Pass recent turns so follow-up questions ("what about
                    # X instead?") are understandable -- answers still stay
                    # grounded in retrieved context, not prior answers.
                    recent_history = [
                        {"question": t["question"], "answer": t["result"].answer}
                        for t in st.session_state.chat_history[-3:]
                        if not t.get("error")
                    ]
                    result = generation.answer_question(
                        question, top_k=5, log_to_db=True, history=recent_history
                    )
                    st.session_state.chat_history.append(
                        {
                            "question": question,
                            "result": result,
                            "wall_time": time.time() - t0,
                        }
                    )
                    st.write(result.answer)
                    render_ref_expander(result)
                except Exception as exc:
                    st.error(f"**Something went wrong:** {exc}")
                    tb = traceback.format_exc()
                    with st.expander("Technical details"):
                        st.code(tb)
                    st.session_state.chat_history.append(
                        {
                            "question": question,
                            "error": str(exc),
                            "traceback": tb,
                        }
                    )

    if st.session_state.chat_history:
        if st.button("Clear conversation"):
            st.session_state.chat_history = []
            st.rerun()


# ===========================================================================
# Router
# ===========================================================================

def main() -> None:
    if st.session_state.page == "chat":
        page_chat()
    else:
        page_home()


if __name__ == "__main__":
    main()
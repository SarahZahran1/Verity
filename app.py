from __future__ import annotations

import json
import time
import traceback
from contextlib import contextmanager
from pathlib import Path

import streamlit as st


import backend_client

st.set_page_config(
    page_title="Verity",
    page_icon="V",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = r"""
<style>
/* ---------- Global ---------- */
:root {
    --bg: #09090f;
    --panel: #10111a;
    --panel-2: #141522;
    --border: #25283a;
    --text: #f4f4f7;
    --muted: #9a9caf;
    --accent: #7c5cff;
    --accent-2: #5b4bdb;
    --user: #211b3d;
}

.stApp {
    background:
        radial-gradient(circle at 50% -10%, rgba(124, 92, 255, 0.12), transparent 32%),
        linear-gradient(180deg, #09090f 0%, #0a0b12 100%);
    color: var(--text);
}

.block-container {
    max-width: 1180px;
    padding-top: 1.5rem;
    padding-bottom: 7rem;
}

/* ---------- Sidebar ---------- */
section[data-testid="stSidebar"] {
    background: #0d0e15;
    border-right: 1px solid var(--border);
}

section[data-testid="stSidebar"] > div {
    padding-top: 1.1rem;
}

.sidebar-brand {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 1.1rem;
}

.sidebar-logo {
    width: 38px;
    height: 38px;
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: linear-gradient(135deg, #7c5cff, #4d3bc7);
    box-shadow: 0 8px 25px rgba(124, 92, 255, 0.25);
    font-size: 16px;
    font-weight: 800;
    color: #fff;
    letter-spacing: -.02em;
}

.sidebar-title {
    font-size: 1.05rem;
    font-weight: 750;
    color: #fff;
}

.sidebar-subtitle {
    color: var(--muted);
    font-size: 0.72rem;
}

.new-chat button {
    border: 1px solid #5f4de0 !important;
    background: rgba(124, 92, 255, 0.12) !important;
    color: #eeeaff !important;
    border-radius: 10px !important;
}

.history-label {
    color: #77798c;
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: .08em;
    margin: 1.2rem 0 .5rem;
}

/* ---------- Header ---------- */
.chat-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin: .3rem 0 1.5rem;
}

.chat-brand {
    display: flex;
    align-items: center;
    gap: 12px;
}

.chat-logo {
    width: 45px;
    height: 45px;
    border-radius: 14px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: linear-gradient(135deg, #7c5cff, #5140cf);
    box-shadow: 0 10px 30px rgba(124, 92, 255, .22);
    font-size: 19px;
    font-weight: 800;
    color: #fff;
    letter-spacing: -.02em;
}

.chat-title {
    font-size: 1.45rem;
    font-weight: 760;
    line-height: 1.1;
}

.chat-subtitle {
    color: var(--muted);
    font-size: .78rem;
    margin-top: 4px;
}

.status-dot {
    display: inline-block;
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: #55d88a;
    margin-right: 6px;
}

/* ---------- Empty state ---------- */
.empty-state {
    text-align: center;
    padding: 8vh 1rem 5vh;
}

.empty-icon {
    width: 56px;
    height: 56px;
    margin: 0 auto 18px;
    border-radius: 16px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: rgba(124, 92, 255, .12);
    border: 1px solid rgba(124, 92, 255, .25);
    font-size: 20px;
    font-weight: 800;
    color: #c9bfff;
    letter-spacing: -.02em;
}

.empty-title {
    font-size: 1.55rem;
    font-weight: 760;
    margin-bottom: 7px;
}

.empty-subtitle {
    color: var(--muted);
    max-width: 570px;
    margin: auto;
    line-height: 1.6;
}

/* ---------- Messages ---------- */
.message-row {
    display: flex;
    margin: 0 0 1.2rem;
}

.message-row.user {
    justify-content: flex-end;
}

.message-row.assistant {
    justify-content: flex-start;
}

.user-bubble {
    max-width: 72%;
    padding: 11px 15px;
    border-radius: 17px 17px 5px 17px;
    background: linear-gradient(135deg, #241d43, #1d1835);
    border: 1px solid #3b3261;
    color: #f5f2ff;
    line-height: 1.55;
    white-space: pre-wrap;
    overflow-wrap: anywhere;
}

.assistant-card {
    width: min(900px, 92%);
    background: linear-gradient(180deg, rgba(18, 19, 29, .98), rgba(15, 16, 25, .98));
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 18px 20px 15px;
    box-shadow: 0 12px 35px rgba(0, 0, 0, .18);
}

.assistant-head {
    display: flex;
    align-items: center;
    gap: 9px;
    color: #ddd9ff;
    font-size: .78rem;
    font-weight: 650;
    margin-bottom: 10px;
}

.assistant-avatar {
    width: 22px;
    height: 22px;
    border-radius: 7px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: rgba(124, 92, 255, .16);
    border: 1px solid rgba(124, 92, 255, .25);
    font-size: 11px;
    font-weight: 800;
    color: #c9bfff;
}

.assistant-content {
    color: #e9e9ef;
    line-height: 1.68;
}

.assistant-content p:last-child {
    margin-bottom: 0;
}

.meta-row {
    display: flex;
    align-items: center;
    gap: 7px;
    margin-top: 12px;
    color: #707285;
    font-size: .68rem;
}

.intent-badge {
    padding: 3px 7px;
    border-radius: 999px;
    border: 1px solid #303246;
    background: #151622;
}

/* ---------- Sources ---------- */
div[data-testid="stExpander"] {
    border: 1px solid #292c3d !important;
    border-radius: 12px !important;
    background: rgba(12, 13, 21, .7) !important;
}

.source-title {
    color: #bcb8d8;
    font-size: .78rem;
    font-weight: 650;
}

.source-id {
    color: #777a91;
    font-family: monospace;
    font-size: .68rem;
}

/* ---------- Code ---------- */
pre {
    border-radius: 12px !important;
    border: 1px solid #292c3d !important;
}

/* ---------- Buttons ---------- */
.stButton > button {
    border-radius: 10px;
    transition: .15s ease;
}

.stButton > button:hover {
    border-color: #6653dc;
}

/* ---------- Chat input ---------- */
div[data-testid="stChatInput"] {
    background: #10111a;
}

div[data-testid="stChatInput"] textarea {
    background: #11121c !important;
    border: 1px solid #2b2e40 !important;
    border-radius: 15px !important;
    color: #f5f5f7 !important;
}

/* ---------- Home ---------- */
.home-hero {
    text-align: center;
    padding: 8vh 1rem 4vh;
}

.home-logo {
    width: 78px;
    height: 78px;
    margin: 0 auto 20px;
    border-radius: 24px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: linear-gradient(135deg, #7c5cff, #4d3bc7);
    box-shadow: 0 18px 45px rgba(124, 92, 255, .25);
    font-size: 32px;
    font-weight: 800;
    color: #fff;
    letter-spacing: -.02em;
}

.home-title {
    font-size: 2.7rem;
    font-weight: 800;
    letter-spacing: -.03em;
}

.home-subtitle {
    color: var(--muted);
    max-width: 650px;
    margin: 10px auto 0;
    font-size: 1rem;
    line-height: 1.65;
}

.feature-card {
    height: 100%;
    padding: 12px 14px;
    border-radius: 10px;
    border: 1px solid var(--border);
    border-left: 2px solid var(--accent);
    background: rgba(17, 18, 27, .72);
}

.feature-title {
    font-weight: 700;
    font-size: .84rem;
    margin-bottom: 1px;
}

.feature-text {
    color: var(--muted);
    font-size: .72rem;
    line-height: 1.4;
}

/* ---------- Mobile ---------- */
@media (max-width: 700px) {
    .block-container {
        padding-left: .8rem;
        padding-right: .8rem;
    }

    .user-bubble,
    .assistant-card {
        max-width: 94%;
        width: 94%;
    }

    .home-title {
        font-size: 2.1rem;
    }
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)



def init_session_state() -> None:
    defaults = {
        "page": "home",
        "chat_history": [],
        "chat_title": "New conversation",
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


init_session_state()


@contextmanager
def safe_run(spinner_text: str):
    placeholder = st.empty()
    try:
        with placeholder:
            with st.spinner(spinner_text):
                yield
    except Exception as exc:
        st.error("Something went wrong.")
        with st.expander("Technical details"):
            st.code(traceback.format_exc())


def backend_ready() -> bool:
    if backend_client.is_backend_ready():
        return True

    st.error(
        f"Verity API is not reachable at {backend_client.API_BASE_URL}. "
        "Make sure it's running (`fastapi dev api.py`)."
    )
    return False


EVAL_METRIC_INFO = {
    "faithfulness": {
        "label": "Faithfulness",
        "help": (
            "Of the claims made in the generated answer, what fraction is "
            "actually supported by the retrieved context? A low score means "
            "the model is hallucinating — saying things the source documents "
            "don't back up."
        ),
    },
    "answer_relevance": {
        "label": "Answer Relevance",
        "help": (
            "Does the answer actually address what was asked? Measured by "
            "how well the answer's content maps back to the original "
            "question — a technically correct but off-target answer scores "
            "lower here."
        ),
    },
    "context_precision": {
        "label": "Context Precision",
        "help": (
            "Of the chunks retrieved from the knowledge base, what fraction "
            "were actually relevant and ranked appropriately? A low score "
            "means retrieval is pulling in noisy or irrelevant chunks."
        ),
    },
    "context_recall": {
        "label": "Context Recall",
        "help": (
            "Of everything needed to answer the question, how much did "
            "retrieval actually surface? A low score means relevant "
            "information exists in the knowledge base but retrieval missed it."
        ),
    },
}


def get_latest_eval_report():
    """Return (eval_data: dict, filename: str) for the latest evaluation report,
    fetched from the API rather than reading data/eval_results/ directly."""
    if not backend_ready():
        return None, None

    return backend_client.get_latest_eval_report()


def new_chat() -> None:
    st.session_state.chat_history = []
    st.session_state.chat_title = "New conversation"


def recent_history_for_backend() -> list[dict]:
    """
    Convert UI state into the history format expected by generation.py.
    Keep only successful turns because failed UI requests should not become
    conversational context.
    """
    history = []

    for turn in st.session_state.chat_history[-4:]:
        if turn.get("error"):
            continue

        result = turn.get("result")
        if result is None:
            continue

        history.append(
            {
                "question": turn["question"],
                "answer": result.answer,
            }
        )

    return history


def intent_label(intent: str) -> str:
    labels = {
        "new_question": "RAG",
        "follow_up": "Follow-up",
        "ack": "Conversation",
        "off_topic": "Out of scope",
    }
    return labels.get(intent or "", "Assistant")


def render_sources(result) -> None:
   
    citations = getattr(result, "citations", None) or []
    retrieved = getattr(result, "retrieved", None) or []

    if not citations or not retrieved:
        return

    cited_ids = set(citations)
    cited_chunks = [r for r in retrieved if r.chunk_id in cited_ids]

    if not cited_chunks:
        return

    with st.expander(f"Sources · {len(cited_chunks)}"):
        for index, r in enumerate(cited_chunks, start=1):
            citation = getattr(r, "citation", "Unknown source")
            chunk_id = getattr(r, "chunk_id", "unknown")
            text = getattr(r, "parent_text", None) or getattr(r, "text", "")

            st.markdown(
                f"""
                <div class="source-title">
                    {index}. {citation}
                </div>
                <div class="source-id">{chunk_id}</div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(text)

            if index != len(cited_chunks):
                st.divider()


def render_user_message(question: str) -> None:
    st.markdown(
        f"""
        <div class="message-row user">
            <div class="user-bubble">{question}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_assistant_result(result, show_actions: bool = True) -> None:
    answer = getattr(result, "answer", "") or ""
    intent = getattr(result, "intent", "new_question")
    used_rag = bool(getattr(result, "used_rag", False))
    refused = bool(getattr(result, "refused", False))

    st.markdown(
        """
        <div class="message-row assistant">
            <div class="assistant-card">
                <div class="assistant-head">
                    <div class="assistant-avatar">V</div>
                    <span>Verity</span>
                </div>
                <div class="assistant-content">
        """,
        unsafe_allow_html=True,
    )

    # Markdown belongs inside the same visual card.
    if refused:
        st.warning(answer)
    else:
        st.markdown(answer)

    st.markdown(
        f"""
                </div>
                <div class="meta-row">
                    <span class="intent-badge">{intent_label(intent)}</span>
                    {"<span>grounded in knowledge base</span>" if used_rag else "<span>conversation only</span>"}
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_sources(result)

    if show_actions and answer:
        cols = st.columns([1, 1, 1, 8])

        with cols[0]:
            if st.button("Copy", key=f"copy_{id(result)}", help="Copy answer"):
                # Streamlit's native clipboard support differs by version.
                # Keep this lightweight: the browser can still copy from the
                # rendered Markdown. This button is intentionally non-invasive.
                st.toast("Select the answer text and copy it.")

        with cols[1]:
            if st.button("Helpful", key=f"up_{id(result)}", help="Helpful"):
                st.toast("Thanks for the feedback!")

        with cols[2]:
            if st.button("Not helpful", key=f"down_{id(result)}", help="Not helpful"):
                st.toast("Thanks — we'll use that feedback.")


def render_error(turn: dict) -> None:
    st.markdown(
        """
        <div class="message-row assistant">
            <div class="assistant-card">
                <div class="assistant-head">
                    <div class="assistant-avatar">V</div>
                    <span>Verity</span>
                </div>
        """,
        unsafe_allow_html=True,
    )
    st.error(turn.get("error", "Unknown error"))
    with st.expander("Technical details"):
        st.code(turn.get("traceback", ""))
    st.markdown("</div></div>", unsafe_allow_html=True)


def conversation_title() -> str:
    history = st.session_state.chat_history
    if not history:
        return "New conversation"

    first_question = history[0].get("question", "").strip()
    if not first_question:
        return "New conversation"

    # Short title for sidebar.
    title = " ".join(first_question.split())
    return title[:34] + ("…" if len(title) > 34 else "")



def render_sidebar() -> None:
    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-brand">
                <div class="sidebar-logo">V</div>
                <div>
                    <div class="sidebar-title">Verity</div>
                    <div class="sidebar-subtitle">Enterprise knowledge assistant</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="new-chat">', unsafe_allow_html=True)
        if st.button("+ New Chat", use_container_width=True, type="primary"):
            new_chat()
            st.session_state.page = "chat"
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="history-label">Conversation</div>', unsafe_allow_html=True)

        if st.session_state.chat_history:
            st.caption(conversation_title())

            if st.button("Clear conversation", use_container_width=True):
                new_chat()
                st.rerun()
        else:
            st.caption("No messages yet")

        st.divider()

        st.markdown("**Supported knowledge**")
        st.caption("Kubernetes · Company policies · Support topics")

        if st.session_state.page == "chat":
            if st.button("Home", use_container_width=True):
                st.session_state.page = "home"
                st.rerun()




def page_home() -> None:
    st.markdown(
        """
        <div class="home-hero">
            <div class="home-logo">V</div>
            <div class="home-title">Verity</div>
            <div class="home-subtitle">
                A grounded enterprise AI assistant that answers questions
                from your internal knowledge base instead of making things up.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
            <div class="feature-card">
                <div>
                    <div class="feature-title">Grounded answers</div>
                    <div class="feature-text">Sourced from your knowledge base.</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
            <div class="feature-card">
                <div>
                    <div class="feature-title">Conversational</div>
                    <div class="feature-text">Remembers context across follow-ups.</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            """
            <div class="feature-card">
                <div>
                    <div class="feature-title">Guarded</div>
                    <div class="feature-text">Refuses instead of guessing.</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")
    eval_data, source = get_latest_eval_report()

    c1, c2, c3 = st.columns([0.3, 5, 0.3])
    with c2:
        with st.container(border=True):
            if eval_data is not None:
                overall = eval_data.get("overall", {}) or {}
                n_samples = eval_data.get("n_samples")
                refusal_acc = eval_data.get("refusal_accuracy_on_adversarial_set")
                by_tier = eval_data.get("by_tier", {}) or {}

                header_col, badge_col = st.columns([3, 1])
                with header_col:
                    st.markdown("### Latest evaluation")
                    st.caption(
                        f"{n_samples} questions evaluated · {source}"
                        if n_samples is not None
                        else source
                    )
                with badge_col:
                    if refusal_acc is not None:
                        st.metric(
                            "Refusal accuracy",
                            f"{refusal_acc:.0%}",
                            help=(
                                "On adversarial / out-of-scope questions "
                                "(e.g. asking for private data or things "
                                "outside the knowledge base), the fraction "
                                "the assistant correctly refused instead of "
                                "guessing."
                            ),
                        )

                st.write("")

                # Average of all four core RAG metrics across the eval set.
                metric_cols = st.columns(4)
                for col, (key, info) in zip(metric_cols, EVAL_METRIC_INFO.items()):
                    value = overall.get(key)
                    with col:
                        st.metric(
                            info["label"],
                            f"{value:.2f}" if value is not None else "—",
                            help=info["help"],
                        )

                st.write("")
                with st.expander("What do these metrics mean?"):
                    for key, info in EVAL_METRIC_INFO.items():
                        st.markdown(f"**{info['label']}** — {info['help']}")

                if by_tier:
                    with st.expander("Breakdown by question tier"):
                        def fmt(v):
                            return f"{v:.2f}" if isinstance(v, (int, float)) else "—"

                        rows = []
                        for tier_name, tier_scores in by_tier.items():
                            rows.append(
                                {
                                    "Tier": tier_name.capitalize(),
                                    "n": tier_scores.get("n"),
                                    "Faithfulness": fmt(tier_scores.get("faithfulness")),
                                    "Answer Relevance": fmt(tier_scores.get("answer_relevance")),
                                    "Context Precision": fmt(tier_scores.get("context_precision")),
                                    "Context Recall": fmt(tier_scores.get("context_recall")),
                                }
                            )
                        st.dataframe(rows, use_container_width=True, hide_index=True)
            else:
                st.markdown("### Latest evaluation")
                st.caption("No evaluation report found yet.")

            st.write("")
            if st.button(
                "Start Chat",
                type="primary",
                use_container_width=True,
            ):
                st.session_state.page = "chat"
                st.rerun()



def page_chat() -> None:
    st.markdown(
        """
        <div class="chat-header">
            <div class="chat-brand">
                <div class="chat-logo">V</div>
                <div>
                    <div class="chat-title">Verity</div>
                    <div class="chat-subtitle">
                        <span class="status-dot"></span>
                        Grounded enterprise assistant
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not backend_ready():
        return

    if not st.session_state.chat_history:
        st.markdown(
            """
            <div class="empty-state">
                <div class="empty-icon">V</div>
                <div class="empty-title">How can I help?</div>
                <div class="empty-subtitle">
                    Ask about Kubernetes, company policies, subscriptions,
                    accounts, support topics, or anything else contained in
                    the Verity knowledge base.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    
    for turn in st.session_state.chat_history:
        render_user_message(turn["question"])

        if turn.get("error"):
            render_error(turn)
        else:
            render_assistant_result(turn["result"], show_actions=False)

    # Composer.
    question = st.chat_input("Message Verity…")

    if not question or not question.strip():
        return

    question = question.strip()

    render_user_message(question)

    # Calls the FastAPI backend's /chat endpoint (routes chit-chat /
    # follow-up / off-topic / new-question), instead of calling
    # generation.handle_message() in-process.
    with st.spinner("Verity is thinking…"):
        try:
            t0 = time.perf_counter()

            recent_history = recent_history_for_backend()

            result = backend_client.chat(
                question=question,
                history=recent_history,
                top_k=5,
                log_to_db=True,
            )

            elapsed = time.perf_counter() - t0

            st.session_state.chat_history.append(
                {
                    "question": question,
                    "result": result,
                    "wall_time": elapsed,
                }
            )

            render_assistant_result(result, show_actions=True)

            # Rerun so the newly added turn becomes part of the stable
            # conversation rendered on the next Streamlit execution.
            # This also keeps Streamlit's state predictable.
            st.rerun()

        except Exception as exc:
            tb = traceback.format_exc()

            st.session_state.chat_history.append(
                {
                    "question": question,
                    "error": str(exc),
                    "traceback": tb,
                }
            )

            render_error(st.session_state.chat_history[-1])




def main() -> None:
    render_sidebar()

    if st.session_state.page == "chat":
        page_chat()
    else:
        page_home()


if __name__ == "__main__":
    main()
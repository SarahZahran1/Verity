from __future__ import annotations
import logging
import os

EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
EMBEDDING_DIM = 768
EMBEDDING_DEVICE = os.environ.get("DOCUMIND_EMBEDDING_DEVICE", "cuda")
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

# Paths 
CHUNKS_PATH = os.environ.get("DOCUMIND_CHUNKS_PATH", "data/processed/chunks_all.jsonl")
PARENTS_PATH = os.environ.get("DOCUMIND_PARENTS_PATH", "data/processed/parents_all.jsonl")
EMBEDDINGS_CACHE_PATH = os.environ.get(
    "DOCUMIND_EMBEDDINGS_PATH", "data/processed/embeddings_bge_base.npy"
)

# Qdrant
QDRANT_URL = os.environ.get("DOCUMIND_QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.environ.get("DOCUMIND_QDRANT_API_KEY")  
QDRANT_COLLECTION_DENSE_ONLY = os.environ.get(
    "DOCUMIND_QDRANT_DENSE_ONLY_COLLECTION", "documind_chunks"
)
QDRANT_COLLECTION = os.environ.get("DOCUMIND_QDRANT_HYBRID_COLLECTION", "documind_chunks_hybrid")
DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"
SPARSE_MODEL = os.environ.get("DOCUMIND_SPARSE_MODEL", "Qdrant/bm25")
EMBED_BATCH_SIZE = int(os.environ.get("DOCUMIND_EMBED_BATCH_SIZE", "32"))
QDRANT_UPSERT_BATCH_SIZE = int(os.environ.get("DOCUMIND_QDRANT_UPSERT_BATCH_SIZE", "256"))


# Retrieval 
RRF_K = int(os.environ.get("DOCUMIND_RRF_K", "60"))
PREFETCH_LIMIT = int(os.environ.get("DOCUMIND_PREFETCH_LIMIT", "40"))
FUSED_TOP_N = int(os.environ.get("DOCUMIND_FUSED_TOP_N", "20"))
FINAL_TOP_K = int(os.environ.get("DOCUMIND_FINAL_TOP_K", "5"))
RERANK_MODEL = os.environ.get("DOCUMIND_RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

LOG_LEVEL = os.environ.get("DOCUMIND_LOG_LEVEL", "INFO")
ENABLE_TIMING = os.environ.get("DOCUMIND_ENABLE_TIMING", "1") not in ("0", "false", "False")


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(name)s] %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
    logger.propagate = False
    return logger



# Generation 
GENERATION_NUM_CTX = 4096
THINKING_ENABLED = False
PARENT_TEXT_MAX_CHARS = 6000

# OpenRouter 
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

GENERATOR_MODEL = os.environ.get("DOCUMIND_GENERATOR_MODEL", "deepseek/deepseek-chat-v3")
JUDGE_MODEL = os.environ.get("DOCUMIND_JUDGE_MODEL", "deepseek/deepseek-chat-v3")
# Intent router (new question / follow-up / ack / greeting / off-topic).
# Defaults to the same model as generation -- override independently if you
# want a cheaper/faster model just for routing.
ROUTER_MODEL = os.environ.get("DOCUMIND_ROUTER_MODEL", GENERATOR_MODEL)
GENERATION_TEMPERATURE = 0.1
JUDGE_TEMPERATURE = 0.0
GENERATION_MAX_TOKENS = 6000
LLM_TIMEOUT_S = 60

# Refusal 
REFUSAL_RERANK_THRESHOLD = float(os.environ.get("DOCUMIND_REFUSAL_THRESHOLD", "-9.64"))

REFUSAL_MESSAGE = (
    "I looked, but I couldn't find anything in the knowledge base that "
    "answers this specifically. The knowledge base covers Kubernetes "
    "documentation, company policies, and support tickets — if your "
    "question fits one of those areas, try rephrasing it or adding a bit "
    "more detail and I'll take another look."
)

# Scope
SCOPE_KEYWORDS = {
    "docs": [
        "kubernetes", "k8s", "kubectl", "pod", "node", "namespace",
        "cluster", "deployment", "container", "helm", "ingress",
        "service", "kubelet", "cpu manager", "rbac", "secret",
        "configmap", "volume", "scheduler", "admission",
        "webhook", "taint", "toleration", "statefulset",
    ],
    "policy": [
        "policy", "pto", "expense", "remote work",
        "security policy", "code of conduct",
        "reimbursement", "vacation", "leave",
        "benefits", "onboarding", "offboarding",
        "compliance",
    ],
    "support": [
        "ticket", "support", "refund", "account",
        "billing", "subscription", "login",
        "password reset", "invoice", "cancel",
    ],
}

SCOPE_EMBEDDING_THRESHOLD = 0.25

OUT_OF_SCOPE_MESSAGE = (
    "That's outside what I can help with here — I'm scoped to Kubernetes "
    "documentation, company policy documents, and support tickets. Feel "
    "free to ask me something in one of those areas!"
)

# Logging
INFERENCE_LOG_DB_PATH = "data/processed/inference_logs.sqlite3"

# Evaluation 
GOLD_EVAL_PATH = "data/gold_eval/gold_qa_set.jsonl"
EVAL_RESULTS_DIR = "data/eval_results"


def require_openrouter_key() -> str:
    if not OPENROUTER_API_KEY:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Export it before running any "
            "generation or evaluation command, e.g.:\n"
            "  export OPENROUTER_API_KEY=sk-or-v1-...\n"
            "(The project no longer ships a hardcoded fallback key.)"
        )
    return OPENROUTER_API_KEY
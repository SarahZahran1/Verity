from __future__ import annotations
import logging
import os

from Retrieval import config as retrieval_config

GENERATION_NUM_CTX = 4096
THINKING_ENABLED = False
PARENT_TEXT_MAX_CHARS = 6000

# ---------- Retrieval ----------
FINAL_TOP_K = retrieval_config.FINAL_TOP_K
get_logger = retrieval_config.get_logger
LOG_LEVEL = retrieval_config.LOG_LEVEL

# ---------- OpenRouter ----------
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "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")

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

# يمكنك تغيير الموديل لأي موديل موجود على OpenRouter
GENERATOR_MODEL = "deepseek/deepseek-chat-v3"
JUDGE_MODEL = "deepseek/deepseek-chat-v3"

GENERATION_TEMPERATURE = 0.1
JUDGE_TEMPERATURE = 0.0

GENERATION_MAX_TOKENS = 1536
LLM_TIMEOUT_S = 120

# ---------- Refusal ----------
REFUSAL_RERANK_THRESHOLD = -9.64

REFUSAL_MESSAGE = (
    "I don't have information on that in the knowledge base."
)

# ---------- Scope ----------
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
    "That question is outside DocuMind's supported domains "
    "(Kubernetes documentation, company policy documents, and support tickets)."
)

# ---------- Logging ----------
INFERENCE_LOG_DB_PATH = "data/processed/inference_logs.sqlite3"

# ---------- Evaluation ----------
GOLD_EVAL_PATH = "data/gold_eval/gold_qa_set.jsonl"
EVAL_RESULTS_DIR = "data/eval_results"


def get_generation_logger(name: str) -> logging.Logger:
    return get_logger(name)
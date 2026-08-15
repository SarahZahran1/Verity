

from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Any

import requests

API_BASE_URL = os.environ.get("VERITY_API_URL", "http://127.0.0.1:8000")
REQUEST_TIMEOUT_S = 60


class BackendError(Exception):
    """Raised when the API is unreachable or returns an error response."""


def _to_namespace(obj: Any) -> Any:
    """Recursively convert dicts/lists from JSON into SimpleNamespace objects
    so callers can use attribute access (obj.field) like before."""
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _to_namespace(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_namespace(v) for v in obj]
    return obj


def is_backend_ready() -> bool:
    try:
        resp = requests.get(f"{API_BASE_URL}/health", timeout=3)
        return resp.status_code == 200
    except requests.RequestException:
        return False


def chat(question: str, history: list[dict], top_k: int = 5, log_to_db: bool = True):
    """Calls POST /chat. Returns a SimpleNamespace mirroring generation.Answer."""
    try:
        resp = requests.post(
            f"{API_BASE_URL}/chat",
            json={
                "question": question,
                "history": history,
                "top_k": top_k,
                "log_to_db": log_to_db,
            },
            timeout=REQUEST_TIMEOUT_S,
        )
    except requests.RequestException as exc:
        raise BackendError(f"Could not reach Verity API at {API_BASE_URL}: {exc}") from exc

    if resp.status_code != 200:
        raise BackendError(f"API error {resp.status_code}: {resp.text}")

    return _to_namespace(resp.json())


def get_latest_eval_report() -> tuple[dict | None, str | None]:
    """Calls GET /eval/latest. Returns (report_dict, filename) or (None, None)."""
    try:
        resp = requests.get(f"{API_BASE_URL}/eval/latest", timeout=10)
    except requests.RequestException:
        return None, None

    if resp.status_code != 200:
        return None, None

    data = resp.json()
    # The API returns the raw report; there's no separate filename field, so
    # we surface a generic label instead of the on-disk filename.
    return data, "latest evaluation report"
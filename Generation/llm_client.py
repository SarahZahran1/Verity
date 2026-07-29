from __future__ import annotations

import json
import re
import time

import requests

from . import config

log = config.get_generation_logger("llm_client")


class LLMError(RuntimeError):
    pass

def _call_openrouter(
    model: str,
    prompt: str,
    system: str | None,
    temperature: float,
    _retries: int = 3,
):
    t0 = time.perf_counter()

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": config.GENERATION_MAX_TOKENS,
    }

    headers = {
        "Authorization": f"Bearer {config.OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://documind.local",
        "X-Title": "DocuMind",
    }

    try:
        resp = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            json=payload,
            headers=headers,
            timeout=config.LLM_TIMEOUT_S,
        )
    except requests.RequestException as e:
        raise LLMError(f"Could not reach OpenRouter: {e}") from e

    if resp.status_code == 429 and _retries > 0:
        time.sleep(5)
        return _call_openrouter(
            model,
            prompt,
            system,
            temperature,
            _retries=_retries - 1,
        )

    if not resp.ok:
        try:
            body = resp.json()
            msg = body.get("error", {}).get("message", resp.text)
        except Exception:
            msg = resp.text
        raise LLMError(msg)

    data = resp.json()
    text = data["choices"][0]["message"]["content"]

    latency = time.perf_counter() - t0
    return text, latency


_THINK_TAG_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _strip_thinking(text: str) -> str:
    return _THINK_TAG_RE.sub("", text).strip()


def generate(prompt: str, system: str | None = None):
    answer, latency =_call_openrouter(
        model=config.GENERATOR_MODEL,
        prompt=prompt,
        system=system,
        temperature=config.GENERATION_TEMPERATURE,
    )
    return _strip_thinking(answer), latency


def judge(prompt: str):
    total_latency = 0.0

    for _ in range(2):
        text, latency =_call_openrouter(
            model=config.JUDGE_MODEL,
            prompt=prompt,
            system=None,
            temperature=config.JUDGE_TEMPERATURE,
        )
        total_latency += latency

        try:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                return json.loads(match.group(0)), total_latency
        except Exception:
            pass

    raise LLMError("Judge model did not return valid JSON.")
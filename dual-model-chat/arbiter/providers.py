"""
Model providers. Everything goes through OpenAI-compatible chat endpoints, so
NVIDIA NIM, Groq, Hugging Face, OpenRouter, Ollama, etc. all work by changing
base_url + api_key + model name via environment variables.

Players in the game:
  A      first answering model
  B      second answering model
  J      judge (defaults to A's provider/model; set JUDGE_* to use a 3rd model)
"""

from __future__ import annotations

import json
import os
import re

from openai import OpenAI

NIM_BASE = "https://integrate.api.nvidia.com/v1"

CFG: dict[str, dict] = {
    "A": {
        "base_url": os.getenv("MODEL_A_BASE_URL", NIM_BASE),
        "api_key": os.getenv("MODEL_A_API_KEY") or os.getenv("NVIDIA_API_KEY", ""),
        "model": os.getenv("MODEL_A_NAME", "nvidia/llama-3.3-nemotron-super-49b-v1.5"),
        "label": os.getenv("MODEL_A_LABEL", "Nemotron"),
    },
    "B": {
        "base_url": os.getenv("MODEL_B_BASE_URL", NIM_BASE),
        "api_key": os.getenv("MODEL_B_API_KEY") or os.getenv("NVIDIA_API_KEY", ""),
        "model": os.getenv("MODEL_B_NAME", "openai/gpt-oss-120b"),
        "label": os.getenv("MODEL_B_LABEL", "GPT-OSS"),
    },
}
CFG["J"] = {
    "base_url": os.getenv("JUDGE_BASE_URL", CFG["A"]["base_url"]),
    "api_key": os.getenv("JUDGE_API_KEY") or CFG["A"]["api_key"],
    "model": os.getenv("JUDGE_NAME", CFG["A"]["model"]),
    "label": "Judge",
}

_clients: dict[str, OpenAI] = {}


def _client(key: str) -> OpenAI:
    if key not in _clients:
        c = CFG[key]
        _clients[key] = OpenAI(base_url=c["base_url"], api_key=c["api_key"])
    return _clients[key]


class ProviderError(RuntimeError):
    pass


def chat(key: str, messages: list[dict], temperature: float = 0.2,
         max_tokens: int = 1500) -> str:
    """One chat completion against player `key`. Raises ProviderError."""
    c = CFG[key]
    if not c["api_key"]:
        raise ProviderError(f"{c['label']}: no API key configured")
    try:
        resp = _client(key).chat.completions.create(
            model=c["model"], messages=messages,
            temperature=temperature, max_tokens=max_tokens,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:  # noqa: BLE001 - surface any SDK/network failure
        raise ProviderError(f"{c['label']}: {type(e).__name__}: {e}") from e


def chat_json(key: str, messages: list[dict], temperature: float = 0.0,
              max_tokens: int = 2000) -> dict:
    """chat() + tolerant JSON extraction (strips fences / leading prose)."""
    raw = chat(key, messages, temperature, max_tokens)
    txt = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.S)
    try:
        return json.loads(txt)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", txt, flags=re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
    raise ProviderError(f"{CFG[key]['label']}: response was not valid JSON: {raw[:200]!r}")

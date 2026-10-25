"""Small, server-side OpenAI-compatible chat adapter for research education.

The application intentionally keeps the drug-severity prediction route separate
and approval-gated. This module never loads the DDI-GUARD model.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


class AssistantConfigurationError(RuntimeError):
    pass


class AssistantRateLimitError(RuntimeError):
    def __init__(self, message: str, retry_after_seconds: float | None = None):
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds


class AssistantUpstreamError(RuntimeError):
    pass


class AssistantTimeoutError(RuntimeError):
    pass


def _load_project_dotenv(path: Path | None = None) -> None:
    """Load supported settings from the project .env without overriding process env.

    This intentionally avoids a runtime dependency on python-dotenv and never logs
    values. Only the assistant settings are read from the file.
    """
    dotenv = path or Path(__file__).resolve().parents[1] / ".env"
    allowed = {
        "DDI_GUARD_AI_PROVIDER",
        "DDI_GUARD_AI_MODEL",
        "DDI_GUARD_AI_BASE_URL",
        "DDI_GUARD_AI_API_KEY",
        "OPENAI_API_KEY",
    }
    try:
        lines = dotenv.read_text(encoding="utf-8").splitlines()
    except OSError:
        return
    for line in lines:
        item = line.strip()
        if not item or item.startswith("#") or "=" not in item:
            continue
        name, value = item.split("=", 1)
        name = name.strip()
        if name not in allowed or name in os.environ:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ[name] = value


_load_project_dotenv()


@dataclass(frozen=True)
class ProviderConfig:
    provider: str
    model: str
    base_url: str
    api_key: str


def provider_config() -> ProviderConfig | None:
    provider = os.environ.get("DDI_GUARD_AI_PROVIDER", "openai").strip().casefold()
    if provider not in {"openai", "openai_compatible"}:
        raise AssistantConfigurationError("DDI_GUARD_AI_PROVIDER must be 'openai' or 'openai_compatible'.")
    key = os.environ.get("DDI_GUARD_AI_API_KEY", "").strip() or os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        return None
    model = os.environ.get("DDI_GUARD_AI_MODEL", "gpt-4.1-mini").strip()
    if not model or len(model) > 120:
        raise AssistantConfigurationError("DDI_GUARD_AI_MODEL must be a non-empty model name of at most 120 characters.")
    if provider == "openai":
        base = "https://api.openai.com/v1"
    else:
        base = os.environ.get("DDI_GUARD_AI_BASE_URL", "").strip().rstrip("/")
        parsed = urlparse(base)
        local_http = parsed.scheme == "http" and parsed.hostname in {"localhost", "127.0.0.1", "::1"}
        if not parsed.hostname or not (parsed.scheme == "https" or local_http):
            raise AssistantConfigurationError("OpenAI-compatible base URL must use HTTPS (HTTP is allowed only for localhost).")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise AssistantConfigurationError("Base URL must not contain credentials, query parameters, or fragments.")
    return ProviderConfig(provider, model, base, key)


def assistant_status() -> dict[str, Any]:
    try:
        config = provider_config()
    except AssistantConfigurationError as exc:
        return {"configured": False, "provider": os.environ.get("DDI_GUARD_AI_PROVIDER", "openai"),
                "model": None, "error": str(exc)}
    if config is None:
        return {"configured": False,
                "provider": os.environ.get("DDI_GUARD_AI_PROVIDER", "openai").strip().casefold(),
                "model": os.environ.get("DDI_GUARD_AI_MODEL", "gpt-4.1-mini"),
                "error": "Set DDI_GUARD_AI_API_KEY or OPENAI_API_KEY in the backend environment."}
    return {"configured": True, "provider": config.provider, "model": config.model, "error": None}


def _retry_after_seconds(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        try:
            return max(0.0, parsedate_to_datetime(value).timestamp() - time.time())
        except (TypeError, ValueError, OverflowError):
            return None


def chat_completion(messages: list[dict[str, str]], *, timeout_seconds: float = 25.0,
                    max_tokens: int = 900, retry_rate_limit: bool = True) -> str:
    config = provider_config()
    if config is None:
        raise AssistantConfigurationError("AI assistant is not configured. Set DDI_GUARD_AI_API_KEY or OPENAI_API_KEY on the backend.")
    body = json.dumps({"model": config.model, "messages": messages,
                       "temperature": 0.2, "max_tokens": max(1, min(int(max_tokens), 900))}).encode("utf-8")
    request = Request(config.base_url + "/chat/completions", data=body, method="POST", headers={
        "Authorization": "Bearer " + config.api_key,
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": "DDI-GUARD-academic-research-demo/1.0",
    })
    attempts = 2 if retry_rate_limit else 1
    for attempt in range(attempts):
        try:
            with urlopen(request, timeout=timeout_seconds) as response:
                payload = json.loads(response.read(1_000_001))
            content = payload["choices"][0]["message"]["content"]
            if not isinstance(content, str) or not content.strip():
                raise AssistantUpstreamError("AI provider returned an empty response.")
            return content.strip()[:12000]
        except HTTPError as exc:
            retry_after = _retry_after_seconds(exc.headers.get("Retry-After") if exc.headers else None)
            if exc.code == 429 and attempt == 0 and retry_rate_limit:
                delay = min(2.0, retry_after) if retry_after is not None else 0.8
                time.sleep(delay)
                continue
            if exc.code == 429:
                raise AssistantRateLimitError("AI provider is rate-limiting requests.", retry_after) from None
            if exc.code in {401, 403}:
                raise AssistantConfigurationError("AI provider rejected its configured credentials or project access.") from None
            raise AssistantUpstreamError(f"AI provider request failed with HTTP {exc.code}.") from None
        except (TimeoutError, OSError) as exc:
            if isinstance(exc, TimeoutError):
                raise AssistantTimeoutError("AI provider timed out. Please retry.") from None
            raise AssistantUpstreamError("Could not reach the configured AI provider.") from None
        except (KeyError, IndexError, TypeError, json.JSONDecodeError):
            raise AssistantUpstreamError("AI provider returned an unexpected response format.") from None
        except URLError as exc:
            if isinstance(exc.reason, TimeoutError):
                raise AssistantTimeoutError("AI provider timed out. Please retry.") from None
            raise AssistantUpstreamError("Could not reach the configured AI provider.") from None
    raise AssistantRateLimitError("AI provider is rate-limiting requests.")


SYSTEM_PROMPT = """You are the DDI-GUARD Research Assistant for academic education and software research.
Clearly distinguish general AI-generated explanation from any supplied retrieved project/source facts. Never call your output a DDI-GUARD model prediction. No model prediction is available in this chat. If supplied context contains a DDInter annotation, describe it only as a dataset label, not a patient outcome. DailyMed excerpts are general label evidence, not pair-specific proof; ChEMBL records are bioactivity evidence, not proof of a drug-drug mechanism. Cite only source labels and URLs supplied in the context; do not invent citations, metrics, files, or source claims. State uncertainty and missing information. Do not give individualized medical advice or recommend starting, stopping, or changing medications or doses; direct such questions to a qualified clinician. This system is not clinically validated."""

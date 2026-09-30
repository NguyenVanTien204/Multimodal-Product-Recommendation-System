from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx

log = logging.getLogger(__name__)

RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class LLMUnavailable(RuntimeError):
    """Raised when no LLM is configured or the provider call failed."""


class LLM(Protocol):
    name: str

    @property
    def enabled(self) -> bool: ...

    def status(self) -> dict[str, Any]: ...

    async def complete(self, system: str, user: str, *, temperature: float = 0.2, max_tokens: int = 700, json_mode: bool = False) -> str: ...


@dataclass
class OpenAICompatLLM:
    """Client for any OpenAI-compatible ``/chat/completions`` endpoint.

    Tested target: Google Gemini through its OpenAI-compatibility layer::

        RAG_LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai
        RAG_LLM_MODEL=gemini-3.5-flash-lite        # or gemini-3.5-flash
        RAG_LLM_API_KEY=<your Gemini API key>
        RAG_LLM_REASONING_EFFORT=low               # keeps "thinking" tokens small and latency low

    Also works with Ollama (``http://host:11434/v1``), vLLM, OpenAI, OpenRouter, ...

    Robustness, because a shop chatbot must not hang on a flaky third-party API:
    * transient errors (429/5xx, timeouts) are retried with backoff (honouring Retry-After);
    * a circuit breaker skips the LLM for `cooldown_s` after `breaker_threshold` consecutive
      failures, so an exhausted quota costs one fast failure instead of a timeout per message;
    * callers catch `LLMUnavailable` and fall back to deterministic template answers
      ("core service must work without the LLM", AGENT.md).
    """

    base_url: str = ""
    model: str = ""
    api_key: str = ""
    timeout_s: float = 30.0
    reasoning_effort: str = ""  # "", "minimal"/"low"/"medium"/"high" (Gemini maps it to its thinking level/budget)
    max_tokens_floor: int = 1024  # reasoning models spend part of max_tokens on thinking; never send less than this
    max_retries: int = 2
    breaker_threshold: int = 3
    cooldown_s: float = 30.0
    transport: Any = None  # injected in tests (httpx.MockTransport)

    _consecutive_failures: int = field(default=0, init=False, repr=False)
    _open_until: float = field(default=0.0, init=False, repr=False)
    _last_error: str = field(default="", init=False, repr=False)
    _last_ok_ms: float | None = field(default=None, init=False, repr=False)

    @property
    def name(self) -> str:
        return f"{self.model}@{self.base_url}" if self.enabled else "disabled"

    @property
    def enabled(self) -> bool:
        return bool(self.base_url and self.model)

    def status(self) -> dict[str, Any]:
        """Health information; never contains the API key."""
        if not self.enabled:
            return {"state": "disabled"}
        state = "circuit_open" if time.monotonic() < self._open_until else ("degraded" if self._consecutive_failures else "ok")
        return {
            "state": state,
            "model": self.model,
            "consecutive_failures": self._consecutive_failures,
            "last_error": self._last_error or None,
            "last_success_latency_ms": self._last_ok_ms,
        }

    # ---- request -----------------------------------------------------------------
    def _body(self, system: str, user: str, temperature: float, max_tokens: int, json_mode: bool) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": temperature,
            "max_tokens": max(max_tokens, self.max_tokens_floor),
            "stream": False,
        }
        if self.reasoning_effort:
            body["reasoning_effort"] = self.reasoning_effort
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        return body

    async def _post(self, client: httpx.AsyncClient, body: dict[str, Any]) -> httpx.Response:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return await client.post(f"{self.base_url.rstrip('/')}/chat/completions", json=body, headers=headers)

    async def complete(
        self, system: str, user: str, *, temperature: float = 0.2, max_tokens: int = 700, json_mode: bool = False
    ) -> str:
        if not self.enabled:
            raise LLMUnavailable("LLM is not configured")
        if time.monotonic() < self._open_until:
            raise LLMUnavailable(f"LLM circuit open after repeated failures ({self._last_error})")

        body = self._body(system, user, temperature, max_tokens, json_mode)
        t0 = time.perf_counter()
        error = "unknown error"
        try:
            async with httpx.AsyncClient(timeout=self.timeout_s, transport=self.transport) as client:
                for attempt in range(self.max_retries + 1):
                    try:
                        resp = await self._post(client, body)
                    except httpx.TransportError as exc:  # timeouts, connection resets
                        error = f"{type(exc).__name__}: {exc}"[:200]
                        await self._backoff(attempt, None)
                        continue

                    if resp.status_code == 400 and "response_format" in body:
                        # Some providers reject json mode; the caller parses JSON out of plain text anyway.
                        body.pop("response_format")
                        resp = await self._post(client, body)
                    if resp.status_code in RETRYABLE_STATUS:
                        error = f"HTTP {resp.status_code}"
                        await self._backoff(attempt, resp.headers.get("retry-after"))
                        continue
                    if resp.status_code >= 400:
                        # 4xx other than the above: retrying cannot help (bad key, unknown model, ...)
                        raise LLMUnavailable(f"HTTP {resp.status_code}: {_error_text(resp)}")
                    return self._finish(resp, t0)
                raise LLMUnavailable(f"LLM unavailable after {self.max_retries + 1} attempts ({error})")
        except LLMUnavailable as exc:
            self._record_failure(str(exc))
            raise

    async def _backoff(self, attempt: int, retry_after: str | None) -> None:
        if attempt >= self.max_retries:
            return
        try:
            delay = min(float(retry_after), 5.0) if retry_after else 0.0
        except ValueError:
            delay = 0.0
        await asyncio.sleep(delay or (0.5 * 2**attempt))

    def _finish(self, resp: httpx.Response, t0: float) -> str:
        try:
            choice = resp.json()["choices"][0]
            content = choice["message"].get("content") or ""
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise LLMUnavailable(f"unexpected LLM response: {exc}") from exc
        if not content.strip():
            reason = choice.get("finish_reason")
            hint = " (raise RAG_LLM_MAX_TOKENS or lower RAG_LLM_REASONING_EFFORT)" if reason == "length" else ""
            raise LLMUnavailable(f"LLM returned an empty completion, finish_reason={reason}{hint}")
        self._consecutive_failures, self._last_error = 0, ""
        self._last_ok_ms = round((time.perf_counter() - t0) * 1000)
        return _strip_reasoning(content)

    def _record_failure(self, message: str) -> None:
        self._consecutive_failures += 1
        self._last_error = message[:200]
        if self._consecutive_failures >= self.breaker_threshold:
            self._open_until = time.monotonic() + self.cooldown_s
            log.warning("LLM circuit opened for %.0fs: %s", self.cooldown_s, self._last_error)


def _error_text(resp: httpx.Response) -> str:
    try:
        data = resp.json()
        if isinstance(data, list) and data:
            data = data[0]
        err = data.get("error", data) if isinstance(data, dict) else data
        return str(err.get("message", err) if isinstance(err, dict) else err)[:200]
    except Exception:  # noqa: BLE001
        return resp.text[:200]


def _strip_reasoning(text: str) -> str:
    """Drop ``<think>...</think>`` blocks some reasoning models emit."""
    while "<think>" in text and "</think>" in text:
        start, end = text.index("<think>"), text.index("</think>") + len("</think>")
        text = text[:start] + text[end:]
    return text.strip()


def parse_json_object(text: str) -> dict[str, Any] | None:
    """Best-effort JSON object extraction from a model reply (fenced or chatty)."""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text[text.find("{") :] if "{" in text else text
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        obj = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None

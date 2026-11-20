"""
Single gateway for every LLM call.

- JSON-only responses, parsed and validated here (callers never regex model output).
- Disk cache: an identical prompt is never paid for twice.
- Per-run call budget and a usage ledger broken down by purpose.
- Failures raise LLMError. Nothing silently degrades to an empty string.
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections import defaultdict
from typing import Any, Callable, Dict, Optional

import httpx

from brain.cache import Cache
from brain.settings import PRICES, Settings

log = logging.getLogger("brain.llm")


class LLMError(RuntimeError):
    pass


class LLMBudgetExceeded(LLMError):
    pass


def _extract_json(text: str) -> Any:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.M).strip()
    start = min([i for i in (text.find("{"), text.find("[")) if i >= 0], default=-1)
    if start < 0:
        raise ValueError("no JSON in model output")
    obj, _ = json.JSONDecoder().raw_decode(text[start:])
    return obj


class LLM:
    def __init__(
        self,
        settings: Optional[Settings] = None,
        cache: Optional[Cache] = None,
        transport: Optional[Callable[[str, str, str, float, int], str]] = None,
    ):
        self.settings = settings or Settings()
        self.cache = cache or Cache()
        self._transport = transport or self._http_transport
        self.calls = 0  # paid (non-cached) calls this run
        self.cached_hits = 0
        self.usage: Dict[str, int] = defaultdict(int)  # purpose -> paid calls
        self.tokens = 0
        self._fast_unavailable = False
        self._dead: set = set()  # models refused this run (missing, or capped)
        self._token_caps: Dict[str, int] = {}  # model -> max output tokens it accepts per request
        self._unsupported: Dict[str, set] = {}  # model -> parameters it rejected
        self.cost_usd = 0.0
        self.switches: list = []
        self._client: Optional[httpx.Client] = None

    @property
    def available(self) -> bool:
        return bool(self.settings.llm_api_key) or self._transport != self._http_transport

    # ── transport ────────────────────────────────────────────────────────────
    def _http_transport(self, model: str, system: str, user: str, temperature: float, max_tokens: int) -> str:
        if not self.settings.llm_api_key:
            raise LLMError("No LLM API key configured (set LLM_API_KEY or GROQ_API_KEY).")
        if self._client is None:
            self._client = httpx.Client(timeout=90)
        payload = {
            "model": model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
        }
        effort = getattr(self, "_reasoning", None) or self.settings.llm_reasoning_effort
        if effort and model.startswith(("openai/gpt-oss", "qwen/")):
            payload["reasoning_effort"] = effort
            payload["max_tokens"] = max_tokens + 1024  # reasoning tokens count against the limit
        cap = self._token_caps.get(model)
        if cap:
            payload["max_tokens"] = min(payload["max_tokens"], cap)
        if "api.openai.com" in self.settings.llm_base_url:
            payload["max_completion_tokens"] = payload.pop("max_tokens")  # OpenAI's current name for the limit
        for param in self._unsupported.get(model, ()):
            payload.pop(param, None)
        url = self.settings.llm_base_url.rstrip("/") + "/chat/completions"
        last = ""
        for attempt in range(4):
            try:
                r = self._client.post(
                    url, json=payload, headers={"Authorization": f"Bearer {self.settings.llm_api_key}"}
                )
            except httpx.HTTPError as e:
                last = repr(e)
                time.sleep(2 ** attempt)
                continue
            if r.status_code == 200:
                data = r.json()
                usage = data.get("usage") or {}
                self.tokens += int(usage.get("total_tokens") or 0)
                p_in, p_out = PRICES.get(model, (0.0, 0.0))
                self.cost_usd += (int(usage.get("prompt_tokens") or 0) * p_in
                                  + int(usage.get("completion_tokens") or 0) * p_out) / 1e6
                return data["choices"][0]["message"]["content"] or ""
            last = f"HTTP {r.status_code}: {r.text[:300]}"
            if r.status_code == 429 and "request too large" in r.text.lower():
                # Not a wait-and-retry limit: this model's per-request output cap is lower than we asked for.
                m = re.search(r"Limit (\d+), Requested (\d+)", r.text)
                if m and int(m.group(1)) >= 400 and payload["max_tokens"] > int(m.group(1)):
                    self._token_caps[model] = int(m.group(1))
                    payload["max_tokens"] = int(m.group(1))
                    continue
                raise _ModelUnavailable(f"{model}: request exceeds its output limit")
            if r.status_code == 400 and ("nsupported" in r.text or "not supported" in r.text):
                # Providers differ on parameters (temperature, reasoning_effort, max_tokens): adapt and retry.
                bad = next((p for p in ("temperature", "reasoning_effort", "max_tokens", "response_format")
                            if f"'{p}'" in r.text or f'"{p}"' in r.text or f" {p} " in r.text), None)
                if bad == "max_tokens" and "max_tokens" in payload:
                    payload["max_completion_tokens"] = payload.pop("max_tokens")
                    continue
                if bad and bad in payload:
                    payload.pop(bad)
                    self._unsupported.setdefault(model, set()).add(bad)
                    continue
            if r.status_code == 400 and "json_validate_failed" in r.text and "response_format" in payload:
                payload.pop("response_format")  # strict JSON mode failed: let the model answer, parse it ourselves
                continue
            if r.status_code == 429:
                wait = _retry_after(r)
                if wait > 60:  # a daily/long cap: waiting is pointless, let the caller switch models
                    raise _RateLimited(f"{model} rate-limited for {int(wait)}s")
                time.sleep(min(wait, 30))
                continue
            if r.status_code in (500, 502, 503, 504):
                time.sleep(2 ** (attempt + 1))
                continue
            if r.status_code in (400, 404) and "model" in r.text.lower():
                raise _ModelUnavailable(last)
            break
        raise LLMError(f"LLM request failed: {last}")

    # ── public API ───────────────────────────────────────────────────────────
    def json(
        self,
        *,
        system: str,
        user: str,
        purpose: str,
        tier: str = "fast",
        temperature: float = 0.0,
        max_tokens: int = 1500,
        use_cache: bool = True,
        reasoning: Optional[str] = None,
    ) -> Any:
        """Run one chat completion and return parsed JSON."""
        model = self._pick(tier)
        key = Cache.key(model, system, user, temperature, reasoning)
        if use_cache:
            hit = self.cache.get("llm", key)
            if hit is not None:
                self.cached_hits += 1
                return hit

        if self.settings.replay == "strict":
            raise LLMError(f"replay(strict): no recorded answer for '{purpose}'")
        system_full = system + "\nRespond with a single valid JSON object and nothing else."
        last_err = ""
        current_user = user
        for attempt in range(6):  # room for model fallbacks plus one repair attempt
            if self.calls >= self.settings.llm_max_calls:
                raise LLMBudgetExceeded(
                    f"LLM call budget of {self.settings.llm_max_calls} reached (raise BRAIN_LLM_MAX_CALLS)."
                )
            self.calls += 1
            self.usage[purpose] += 1
            try:
                self._reasoning = reasoning
                raw = self._transport(model, system_full, current_user, temperature, max_tokens)
            except (_ModelUnavailable, _RateLimited) as e:
                self._dead.add(model)
                nxt = self._pick(tier)
                if nxt is None:
                    raise LLMError(f"All configured models are unavailable or rate-limited ({e}). "
                                   "Wait for the limit to reset or set LLM_FALLBACK_MODELS.")
                log.warning("%s; switching to %s for the rest of this run", e, nxt)
                self.switches.append(f"{model} -> {nxt}")
                model = nxt
                self.calls -= 1  # a refused request is not a paid call
                self.usage[purpose] -= 1
                continue
            try:
                parsed = _extract_json(raw)
            except ValueError as e:
                last_err = f"{e}: {raw[:200]!r}"
                # Resending an identical request at temperature 0 fails the same way: show the model its mistake.
                current_user = (user + f"\n\nYour previous reply could not be parsed as JSON ({e}). It began: "
                                f"{raw[:300]!r}\nReply again with ONLY the corrected JSON object, no prose, no code fences.")
                continue
            if use_cache:
                self.cache.set("llm", key, parsed, ttl=30 * 86400)
            return parsed
        raise LLMError(f"LLM returned unparseable output for '{purpose}': {last_err}")

    def _pick(self, tier: str) -> Optional[str]:
        order = [self.settings.llm_model_strong]
        if tier == "fast":
            order.insert(0, self.settings.llm_model_fast)
        order += [m.strip() for m in self.settings.llm_fallback_models.split(",") if m.strip()]
        if tier == "strong":
            order.append(self.settings.llm_model_fast)
        return next((m for m in dict.fromkeys(order) if m not in self._dead), None)

    def report(self) -> Dict[str, Any]:
        return {
            "paid_calls": self.calls,
            "cached_hits": self.cached_hits,
            "tokens": self.tokens,
            "est_cost_usd": round(self.cost_usd, 4),
            "by_purpose": {k: v for k, v in self.usage.items() if v},
            "model_switches": self.switches,
        }


class _ModelUnavailable(LLMError):
    pass


class _RateLimited(LLMError):
    pass


def _retry_after(r) -> float:
    header = r.headers.get("retry-after")
    if header:
        try:
            return float(header)
        except ValueError:
            pass
    m = re.search(r"try again in (?:(\d+)h)?(?:(\d+)m)?([\d.]+)s", r.text)
    if m:
        h, mi, sec = m.groups()
        return int(h or 0) * 3600 + int(mi or 0) * 60 + float(sec)
    return 5.0

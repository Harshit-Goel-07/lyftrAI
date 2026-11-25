import os
import time
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

import litellm
from litellm.caching.caching import Cache

VALID_SIMULATED_STATUSES = {"429", "500", "502", "503"}


class AllProvidersUnavailableError(Exception):
    """Raised when every configured API key fails."""


def init_litellm_cache() -> None:
    if getattr(litellm, "cache", None) is None:
        litellm.cache = Cache(type="local")


def parse_simulate_failure_header(header_value: Optional[str]) -> Dict[str, int]:
    """Parse X-Simulate-Failure header into a mapping of key_id (or '*') -> status code."""
    if not header_value:
        return {}
    failures: Dict[str, int] = {}
    for pair in header_value.split(","):
        pair = pair.strip()
        if "=" not in pair:
            continue
        key_part, status_part = pair.split("=", 1)
        key_id = key_part.strip()
        status_str = status_part.strip()
        if key_id and status_str in VALID_SIMULATED_STATUSES:
            failures[key_id] = int(status_str)
    return failures


def _is_retryable_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int) and (status_code == 429 or 500 <= status_code < 600):
        return True
    err_name = type(exc).__name__.lower()
    retryable_names = (
        "ratelimit",
        "serviceunavailable",
        "internalserver",
        "apiconnection",
        "timeout",
        "badgateway",
        "quota",
    )
    if any(name in err_name for name in retryable_names):
        return True
    msg = str(exc).lower()
    if any(token in msg for token in ("429", "quota", "rate limit", "500", "502", "503", "504")):
        return True
    return True


def generate_chat_completion(
    model: str,
    messages: List[Dict[str, str]],
    api_keys: List[str],
    simulate_failure_header: Optional[str] = None,
    enable_failure_simulation: bool = False,
) -> Tuple[str, Dict[str, Any]]:
    if not api_keys:
        raise AllProvidersUnavailableError("No API keys configured")

    init_litellm_cache()

    simulated_failures: Dict[str, int] = {}
    if enable_failure_simulation and simulate_failure_header:
        simulated_failures = parse_simulate_failure_header(simulate_failure_header)

    start_time = time.perf_counter()
    fallback_attempts = 0

    for idx, api_key in enumerate(api_keys, start=1):
        key_id = f"key_{idx}"

        if "*" in simulated_failures or key_id in simulated_failures:
            fallback_attempts += 1
            continue

        try:
            response = litellm.completion(
                model=model,
                messages=messages,
                api_key=api_key,
                caching=True,
            )
            elapsed_ms = max(1, int((time.perf_counter() - start_time) * 1000))

            hidden_params = getattr(response, "_hidden_params", {}) or {}
            is_cache_hit = bool(hidden_params.get("cache_hit", False))

            reply_text = response.choices[0].message.content or ""
            usage = getattr(response, "usage", None)
            input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
            output_tokens = int(getattr(usage, "completion_tokens", 0) or 0)

            metadata: Dict[str, Any] = {
                "resolution_source": "LLM_CACHE" if is_cache_hit else "LIVE_INFERENCE",
                "model": model,
                "latency_ms": elapsed_ms,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "key_id": None if is_cache_hit else key_id,
                "fallback_attempts": 0 if is_cache_hit else fallback_attempts,
            }
            return reply_text, metadata

        except Exception as exc:
            if _is_retryable_error(exc):
                fallback_attempts += 1
                continue
            raise

    raise AllProvidersUnavailableError("all providers unavailable")

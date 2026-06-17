"""Async client for the Anthropic Claude API."""

import asyncio
import json
import logging
import os
import random
import re
from typing import Callable, Optional

import anthropic

logger = logging.getLogger(__name__)


def _model_rejects_sampling_params(model: str) -> bool:
    """True for Claude models that have DEPRECATED sampling params (temperature,
    top_p, top_k) and return a 400 if they are sent.

    Opus 4.7 and later (4.7, 4.8, any future opus-4-N with N>=7, and opus-5+) no
    longer accept `temperature`; sending it wastes a call on the
    400-then-strip-then-retry path. The version is parsed from the model id so
    newer Opus releases are covered without a code change. Other families (e.g.
    Sonnet 4.6, used for blind-solve) still accept temperature and are
    unaffected. Anything unparseable returns False, leaving the runtime 400-net
    in `generate()` as the backstop.
    """
    m = model.lower()
    if "opus" not in m:
        return False
    match = re.search(r"opus-(\d+)(?:-(\d+))?", m)
    if not match:
        return False
    major = int(match.group(1))
    minor = int(match.group(2) or 0)
    return (major, minor) >= (4, 7)


class LLMClient:
    """Async client for the Anthropic API with batched request support and backoff."""

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        timeout: float = 120.0,
        max_retries: int = 6,
        initial_backoff: float = 2.0,
        max_backoff: float = 60.0,
    ):
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.initial_backoff = initial_backoff
        self.max_backoff = max_backoff
        # Models that rejected `temperature` (e.g. Opus 4.8). Learned at runtime
        # so we stop sending it for those models after the first 400.
        self._no_temperature_models: set[str] = set()

        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Export it before running."
            )
        self._api_key = key
        self._client: Optional[anthropic.AsyncAnthropic] = None

    async def __aenter__(self):
        self._client = anthropic.AsyncAnthropic(
            api_key=self._api_key, timeout=self.timeout
        )
        return self

    async def __aexit__(self, *args):
        if self._client:
            await self._client.close()
            self._client = None

    async def _ensure_client(self):
        if self._client is None:
            self._client = anthropic.AsyncAnthropic(
                api_key=self._api_key, timeout=self.timeout
            )

    @staticmethod
    def _split_messages(messages: list[dict]) -> tuple[Optional[str], list[dict]]:
        """Split OpenAI-style messages into (system, user/assistant messages).

        Anthropic takes the system prompt as a separate top-level parameter.
        """
        system_parts: list[str] = []
        chat: list[dict] = []
        for m in messages:
            if m.get("role") == "system":
                system_parts.append(m.get("content", ""))
            else:
                chat.append({"role": m["role"], "content": m["content"]})
        system = "\n\n".join(p for p in system_parts if p) or None
        return system, chat

    async def generate(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        model: Optional[str] = None,
        on_usage: Optional[Callable[[str, int, int], None]] = None,
    ) -> str:
        """Send a single message request. Returns the text content.

        Args:
            model: Override the client's default model for this call (e.g. to
                route a cheaper validation step to a different model).
            on_usage: Optional callback invoked once per successful call with
                (model, input_tokens, output_tokens) for cost/metrics tracking.
        """
        await self._ensure_client()
        system, chat = self._split_messages(messages)

        use_model = model or self.model
        kwargs = {
            "model": use_model,
            "messages": chat,
            "max_tokens": max_tokens,
        }
        # Sampling params (temperature/top_p/top_k) are DEPRECATED on Opus 4.7+
        # and return a 400. Omit `temperature` up front for those models so we
        # never waste a call on the 400-then-retry path, and also omit it for any
        # model we learned at runtime rejects it. (This client never sends top_p
        # or top_k, so there is nothing extra to strip for them.) All other
        # models — e.g. Sonnet 4.6 for blind-solve — still receive `temperature`.
        if (
            use_model not in self._no_temperature_models
            and not _model_rejects_sampling_params(use_model)
        ):
            kwargs["temperature"] = temperature
        if system is not None:
            kwargs["system"] = system

        attempt = 0
        while True:
            try:
                resp = await self._client.messages.create(**kwargs)
                if on_usage is not None:
                    try:
                        usage = getattr(resp, "usage", None)
                        on_usage(
                            use_model,
                            getattr(usage, "input_tokens", 0) or 0,
                            getattr(usage, "output_tokens", 0) or 0,
                        )
                    except Exception as e:  # never let metrics break generation
                        logger.debug(f"on_usage callback failed: {e}")
                # Concatenate any text blocks in the response.
                parts = []
                for block in resp.content:
                    text = getattr(block, "text", None)
                    if text:
                        parts.append(text)
                result = "".join(parts)
                if not result:
                    # No text block — e.g. a thinking-only or truncated response.
                    # Dump the raw structure so the format is visible in logs.
                    block_types = [
                        getattr(b, "type", type(b).__name__) for b in resp.content
                    ]
                    logger.warning(
                        f"{use_model} returned no text content "
                        f"(stop_reason={getattr(resp, 'stop_reason', None)}, "
                        f"blocks={block_types}). Raw content (first 800 chars): "
                        f"{str(resp.content)[:800]}"
                    )
                return result
            except (
                anthropic.RateLimitError,
                anthropic.APITimeoutError,
                anthropic.APIConnectionError,
                anthropic.InternalServerError,
            ) as e:
                attempt += 1
                if attempt > self.max_retries:
                    logger.error(f"Anthropic request failed after {attempt} attempts: {e}")
                    raise
                delay = min(
                    self.max_backoff,
                    self.initial_backoff * (2 ** (attempt - 1)),
                )
                delay = delay * (0.5 + random.random())  # jitter
                logger.warning(
                    f"Transient error ({type(e).__name__}); retrying in {delay:.1f}s "
                    f"(attempt {attempt}/{self.max_retries})"
                )
                await asyncio.sleep(delay)
            except anthropic.APIStatusError as e:
                status = getattr(e, "status_code", None)
                # Some models reject `temperature` with a 400 (e.g. Opus 4.8).
                # Drop it, remember the model, and retry — not counted as a retry.
                if (
                    status == 400
                    and "temperature" in kwargs
                    and "temperature" in str(e).lower()
                ):
                    kwargs.pop("temperature", None)
                    self._no_temperature_models.add(use_model)
                    logger.warning(
                        f"Model '{use_model}' rejected `temperature`; omitting it "
                        f"for this model and retrying. ({e})"
                    )
                    continue
                if status == 429 or (status is not None and 500 <= status < 600):
                    attempt += 1
                    if attempt > self.max_retries:
                        logger.error(
                            f"Anthropic request failed after {attempt} attempts: {e}"
                        )
                        raise
                    delay = min(
                        self.max_backoff,
                        self.initial_backoff * (2 ** (attempt - 1)),
                    )
                    delay = delay * (0.5 + random.random())
                    logger.warning(
                        f"HTTP {status}; retrying in {delay:.1f}s "
                        f"(attempt {attempt}/{self.max_retries})"
                    )
                    await asyncio.sleep(delay)
                    continue
                logger.error(f"Anthropic API error {status}: {e}")
                raise

    async def generate_json(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        model: Optional[str] = None,
        on_usage: Optional[Callable[[str, int, int], None]] = None,
    ) -> dict:
        """Send a request and parse the response as JSON.

        Handles common issues like markdown code fences around JSON.
        """
        raw = await self.generate(
            messages, temperature, max_tokens, model=model, on_usage=on_usage
        )
        try:
            return parse_json_response(raw)
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(
                f"Failed to parse JSON from {model or self.model} response: {e}. "
                f"Raw response (first 800 chars): {raw[:800]!r}"
            )
            raise

    async def generate_batch(
        self,
        requests: list[dict],
        batch_size: int = 5,
    ) -> list[Optional[str]]:
        """Send multiple requests concurrently in batches.

        Each item in `requests` should be a dict with:
          - messages: list[dict]
          - temperature: float (optional, default 0.7)
          - max_tokens: int (optional, default 2048)

        Returns results in the same order as input.
        """
        results: list[Optional[str]] = [None] * len(requests)

        for batch_start in range(0, len(requests), batch_size):
            batch = requests[batch_start : batch_start + batch_size]
            tasks = [
                self.generate(
                    messages=req["messages"],
                    temperature=req.get("temperature", 0.7),
                    max_tokens=req.get("max_tokens", 2048),
                )
                for req in batch
            ]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            for i, result in enumerate(batch_results):
                idx = batch_start + i
                if isinstance(result, Exception):
                    logger.error(f"Request {idx} failed: {result}")
                    results[idx] = None
                else:
                    results[idx] = result

        return results

    async def generate_json_batch(
        self,
        requests: list[dict],
        batch_size: int = 5,
    ) -> list[Optional[dict]]:
        """Send multiple requests and parse each as JSON."""
        raw_results = await self.generate_batch(requests, batch_size)
        parsed: list[Optional[dict]] = []
        for i, raw in enumerate(raw_results):
            if raw is None:
                parsed.append(None)
                continue
            try:
                parsed.append(parse_json_response(raw))
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"Failed to parse JSON for request {i}: {e}")
                logger.debug(f"Raw response: {raw[:500]}")
                parsed.append(None)
        return parsed

    async def health_check(self) -> bool:
        """Verify the API key works by issuing a trivial request."""
        await self._ensure_client()
        try:
            await self._client.messages.create(
                model=self.model,
                max_tokens=4,
                messages=[{"role": "user", "content": "ping"}],
            )
            return True
        except Exception as e:
            logger.error(f"Anthropic health check failed: {e}")
            return False


_HEX_DIGITS = frozenset("0123456789abcdefABCDEF")
# Control chars that a SINGLE-backslash LaTeX command silently decodes to under
# strict JSON: \text->\t (TAB), \frac->\f (FF), \rho->\r (CR), \beta->\b (BS).
# A real newline (\n) is legitimate content, so it is NOT in this set.
_CORRUPTING_CONTROL = ("\t", "\x0c", "\r", "\x08")


def _repair_json_escapes(text: str) -> str:
    r"""Make backslash-heavy LaTeX survive JSON parsing.

    Generated content is now LaTeX ($K_\text{a}$, $\Delta$, $5.0\ \text{V/cm}$).
    A LaTeX backslash inside a JSON string value is only valid JSON if doubled
    (``\\text``). When the model emits a single backslash it is either an invalid
    escape (``\Delta`` -> ``\D``, bare ``\ ``) that makes ``json.loads`` raise
    ``Invalid \escape``, or — worse, silently — a backslash whose next char is a
    JSON escape letter (``\text`` -> TAB, ``\rho`` -> CR), which parses to a
    control char and mangles the LaTeX without error.

    This repair keeps ONLY genuine JSON escapes — ``\\`` (an escaped-backslash
    pair, consumed as a unit so it is never re-doubled), ``\"``, ``\/``, and
    ``\uXXXX`` (4 hex digits) — and doubles EVERY other backslash. So:
      * correctly escaped ``\\text`` is preserved -> decodes to ``\text``;
      * unescaped ``\text``/``\times``/``\Delta``/bare ``\ `` are doubled ->
        decode to the intended single-backslash LaTeX (renders in KaTeX).
    Backslash-letter sequences are treated as LaTeX, so a genuine ``\n``/``\t``
    control escape in the raw would be doubled; that is acceptable because this
    only runs when a response either failed strict parse or already shows the
    control-char corruption above — well-formed responses are returned by the
    strict pass and never reach here.
    """
    out: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        c = text[i]
        if c == "\\":
            nxt = text[i + 1] if i + 1 < n else ""
            if nxt == "\\":
                # Escaped-backslash pair — keep both, consume as a unit.
                out.append("\\\\")
                i += 2
                continue
            if nxt in '"/':
                out.append(c)
                out.append(nxt)
                i += 2
                continue
            if (
                nxt == "u"
                and i + 6 <= n
                and all(ch in _HEX_DIGITS for ch in text[i + 2 : i + 6])
            ):
                # Genuine \uXXXX unicode escape (not LaTeX like \upsilon).
                out.append(text[i : i + 6])
                i += 6
                continue
            # LaTeX command or otherwise-invalid escape — double the backslash.
            out.append("\\\\")
            i += 1
            continue
        out.append(c)
        i += 1
    return "".join(out)


def _extract_balanced_object(text: str) -> Optional[str]:
    r"""Return the substring from the first top-level ``{`` to its matching ``}``.

    Walks the string tracking JSON string-literal state (respecting ``\"``
    escapes) and brace depth, so a ``}`` that appears inside a string value does
    NOT close the object. Recovers a JSON object embedded in prose — e.g.
    ``I need to verify... { ...valid json... } done`` — that the naive
    first-``{`` / last-``}`` slice would mangle when prose contains stray braces.
    Returns None if there is no ``{`` or no balanced close (e.g. truncated).
    """
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        c = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif c == "\\":
                escaped = True
            elif c == '"':
                in_string = False
            continue
        if c == '"':
            in_string = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _has_corrupting_control(obj) -> bool:
    """True if any string in the parsed object contains a control char that a
    single-backslash LaTeX command (``\\text`` etc.) would have decoded to."""
    if isinstance(obj, str):
        return any(ch in obj for ch in _CORRUPTING_CONTROL)
    if isinstance(obj, dict):
        return any(_has_corrupting_control(v) for v in obj.values())
    if isinstance(obj, list):
        return any(_has_corrupting_control(v) for v in obj)
    return False


def _coerce_to_object(parsed):
    """Unwrap a list-wrapped JSON object so ``RawX(**parsed)`` unpacking works.

    Models occasionally return a single question/passage as a one-element JSON
    ARRAY (``[ {...} ]``) instead of a bare object. Left alone, the schema
    unpacking raises ``argument after ** must be a mapping, not list`` and the
    item is discarded. This recovers it STRUCTURALLY — the content is untouched,
    so the item still goes through the normal adversarial + blind-solve gates:
      * a dict is returned unchanged;
      * a single-element list whose element is a dict is unwrapped to that dict;
      * a longer list uses its FIRST dict element (logging a warning), ignoring
        the rest;
      * an empty list, or a list with no dict element, raises ValueError so
        parsing fails cleanly (no fabrication), exactly as before.
    """
    if isinstance(parsed, dict):
        return parsed
    if isinstance(parsed, list):
        if len(parsed) == 1 and isinstance(parsed[0], dict):
            return parsed[0]
        for item in parsed:
            if isinstance(item, dict):
                logger.warning(
                    f"Parsed a JSON array of {len(parsed)} element(s); using the "
                    f"first object and ignoring the rest."
                )
                return item
        raise ValueError(
            f"Parsed JSON array has no object to unwrap (len={len(parsed)})"
        )
    raise ValueError(f"Parsed JSON is a {type(parsed).__name__}, expected an object")


def parse_json_response(raw: str) -> dict:
    r"""Parse JSON from LLM output, handling code fences and LaTeX backslashes.

    For each candidate substring: try a STRICT parse first so a well-formed
    response (including correctly escaped ``\\text``) is returned untouched. If
    strict parsing fails — or succeeds but yields control-char-corrupted LaTeX
    from single-backslash commands like ``\text`` -> TAB — fall back to
    ``_repair_json_escapes`` and re-parse. Shared by the discrete, science, and
    CARS pipelines.
    """
    text = raw.strip()

    # Strip markdown code fences
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:]  # drop opening fence
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)

    # Candidate substrings in priority order: the whole thing, then the
    # balance-matched {...} object (robust to prose before/after and to a "}"
    # inside a string value), then the naive outermost {...} and [...] slices as
    # further fallbacks.
    candidates = [text]
    balanced = _extract_balanced_object(text)
    if balanced is not None:
        candidates.append(balanced)
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        candidates.append(text[start:end])
    start = text.find("[")
    end = text.rfind("]") + 1
    if start >= 0 and end > start:
        candidates.append(text[start:end])

    for cand in candidates:
        strict_result = None
        try:
            strict_result = json.loads(cand)
        except json.JSONDecodeError:
            strict_result = None
        # Accept the strict result only if it is clean — i.e. no LaTeX command
        # silently decoded into a control char. A list-wrapped object is
        # unwrapped to its dict (see _coerce_to_object). A candidate that parses
        # but is NOT coercible to an object — e.g. the trailing "[...]" slice
        # happens to capture a stray array literal ("[1, 2, 3]", "[]") sitting
        # inside the prose of a stem/explanation — must be SKIPPED so we try the
        # next candidate and ultimately raise the clean "Could not parse" error,
        # not let _coerce_to_object's ValueError escape as a misleading
        # "no object to unwrap (len=N)". So coercion failure here is non-fatal.
        if strict_result is not None and not _has_corrupting_control(strict_result):
            try:
                return _coerce_to_object(strict_result)
            except ValueError:
                pass
        # Strict failed or produced corrupted LaTeX — repair escapes and retry.
        try:
            return _coerce_to_object(json.loads(_repair_json_escapes(cand)))
        except (json.JSONDecodeError, ValueError):
            pass
        # Repair didn't parse; fall back to the (corrupted-but-usable) strict
        # result rather than losing the response entirely.
        if strict_result is not None:
            try:
                return _coerce_to_object(strict_result)
            except ValueError:
                pass

    raise ValueError(f"Could not parse JSON from response: {text[:200]}...")

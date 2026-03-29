"""Async client for vLLM's OpenAI-compatible API."""

import asyncio
import json
import logging
from typing import Optional

import httpx

logger = logging.getLogger(__name__)


class LLMClient:
    """Async client for vLLM server with batched request support."""

    def __init__(self, base_url: str, model: str, timeout: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self):
        self._client = httpx.AsyncClient(timeout=self.timeout)
        return self

    async def __aexit__(self, *args):
        if self._client:
            await self._client.aclose()

    async def _ensure_client(self):
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)

    async def generate(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> str:
        """Send a single chat completion request. Returns the text content."""
        await self._ensure_client()

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        try:
            resp = await self._client.post(
                f"{self.base_url}/chat/completions",
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error {e.response.status_code}: {e.response.text[:500]}")
            raise
        except Exception as e:
            logger.error(f"LLM request failed: {e}")
            raise

    async def generate_json(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> dict:
        """Send a chat completion and parse the response as JSON.

        Handles common issues like markdown code fences around JSON.
        """
        raw = await self.generate(messages, temperature, max_tokens)
        return parse_json_response(raw)

    async def generate_batch(
        self,
        requests: list[dict],
        batch_size: int = 20,
    ) -> list[str]:
        """Send multiple requests concurrently in batches.

        Each item in `requests` should be a dict with:
          - messages: list[dict]
          - temperature: float (optional, default 0.7)
          - max_tokens: int (optional, default 2048)

        Returns results in the same order as input.
        """
        results = [None] * len(requests)

        for batch_start in range(0, len(requests), batch_size):
            batch = requests[batch_start : batch_start + batch_size]
            tasks = []
            for req in batch:
                tasks.append(
                    self.generate(
                        messages=req["messages"],
                        temperature=req.get("temperature", 0.7),
                        max_tokens=req.get("max_tokens", 2048),
                    )
                )
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
        batch_size: int = 20,
    ) -> list[Optional[dict]]:
        """Send multiple requests and parse each as JSON."""
        raw_results = await self.generate_batch(requests, batch_size)
        parsed = []
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
        """Check if the vLLM server is reachable."""
        await self._ensure_client()
        try:
            resp = await self._client.get(f"{self.base_url}/models")
            return resp.status_code == 200
        except Exception:
            return False


def parse_json_response(raw: str) -> dict:
    """Parse JSON from LLM output, handling code fences and common issues."""
    text = raw.strip()

    # Strip markdown code fences
    if text.startswith("```"):
        # Remove first line (```json or ```)
        lines = text.split("\n")
        lines = lines[1:]  # drop opening fence
        # Remove closing fence
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)

    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to find JSON object in the text
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass

    # Try to find JSON array
    start = text.find("[")
    end = text.rfind("]") + 1
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not parse JSON from response: {text[:200]}...")

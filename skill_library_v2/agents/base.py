"""Shared helpers for LLM-backed agents in skill_library_v2.

The :class:`BaseLLMAgent` captures the boring parts — client selection,
tenacity retry around transient errors, robust JSON extraction with a single
corrective retry — so each agent module can focus on its prompt and contract.
"""

from __future__ import annotations

import json
import logging
import re
from typing import TypeVar

from openai import APIError, APITimeoutError, AsyncAzureOpenAI, RateLimitError
from pydantic import BaseModel, ValidationError
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from llm_client import FAST_MODEL, REASONING_MODEL, get_fast_client, get_reasoning_client

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

_FENCE_RE = re.compile(r"```(?:json)?\s*(?P<body>.*?)```", re.DOTALL | re.IGNORECASE)


def _strip_code_fence(raw: str) -> str:
    """Strip a ```json ... ``` fence if the model emitted one despite instructions."""
    m = _FENCE_RE.search(raw)
    return m.group("body").strip() if m else raw.strip()


def parse_llm_json(raw: str, schema: type[T]) -> T:
    """Parse and validate LLM output against a Pydantic schema.

    Raises the underlying :class:`pydantic.ValidationError` or :class:`ValueError`
    on failure; callers decide whether to retry with a corrective message.
    """
    body = _strip_code_fence(raw)
    try:
        data = json.loads(body)
    except json.JSONDecodeError as exc:
        raise ValueError(f"model did not return valid JSON: {exc}") from exc
    return schema.model_validate(data)


class BaseLLMAgent:
    """Base class for LLM-backed agents.

    Subclasses typically override ``tier`` (fast vs reasoning) and implement
    their own ``__call__`` or node coroutine. The helpers here are available
    on the instance or as classmethods.
    """

    #: "fast" → gpt-4o-mini, "reasoning" → o4-mini (see llm_client.py).
    tier: str = "reasoning"

    def __init__(self, *, agent_name: str, prompt_version: str) -> None:
        self.agent_name = agent_name
        self.prompt_version = prompt_version

    # ── Client + model name ──────────────────────────────────────────────
    def get_client(self) -> AsyncAzureOpenAI:
        client = get_reasoning_client() if self.tier == "reasoning" else get_fast_client()
        if client is None:
            raise RuntimeError(
                "Azure OpenAI client unavailable — is AZURE_OPEN_AI_KEY set?"
            )
        return client

    @property
    def model_name(self) -> str:
        return REASONING_MODEL if self.tier == "reasoning" else FAST_MODEL

    # ── Retry wrapper for a single JSON-returning completion call ────────
    async def call_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        schema: type[T],
        temperature: float = 0.2,
        max_output_tokens: int | None = None,
    ) -> T:
        """Call the LLM, expect JSON, validate against ``schema``.

        On a validation failure the agent retries once with a corrective
        system message. Transient OpenAI errors (rate limit, timeout, 5xx)
        get 3 tenacity attempts with exponential backoff.
        """
        client = self.get_client()
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        last_raw: str | None = None
        last_err: Exception | None = None

        for attempt_idx in range(2):  # original + one corrective retry
            try:
                async for attempt in AsyncRetrying(
                    stop=stop_after_attempt(3),
                    wait=wait_exponential(multiplier=1, min=1, max=8),
                    retry=retry_if_exception_type(
                        (RateLimitError, APITimeoutError, APIError)
                    ),
                    reraise=True,
                ):
                    with attempt:
                        kwargs: dict = {
                            "model": self.model_name,
                            "messages": messages,
                            "response_format": {"type": "json_object"},
                        }
                        # o-series reasoning models reject `temperature` / token caps;
                        # pass them only for the fast tier.
                        if self.tier == "fast":
                            kwargs["temperature"] = temperature
                            if max_output_tokens is not None:
                                kwargs["max_tokens"] = max_output_tokens
                        resp = await client.chat.completions.create(**kwargs)
                        last_raw = resp.choices[0].message.content or ""

                return parse_llm_json(last_raw, schema)

            except (ValidationError, ValueError) as exc:
                last_err = exc
                if attempt_idx == 0:
                    logger.warning(
                        "[%s] JSON validation failed; retrying with corrective message.",
                        self.agent_name,
                    )
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                        {"role": "assistant", "content": last_raw or ""},
                        {
                            "role": "user",
                            "content": (
                                "Your previous reply did not match the required schema. "
                                f"Error: {exc}. Re-emit a valid JSON object matching the "
                                "schema in the system prompt. No prose, no code fences."
                            ),
                        },
                    ]
                    continue
                raise

        assert last_err is not None
        raise last_err

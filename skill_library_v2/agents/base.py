"""Shared helpers for LLM-backed agents in skill_library_v2.

The :class:`BaseLLMAgent` captures the boring parts — client selection,
tenacity retry around transient errors, robust JSON extraction with a single
corrective retry — so each agent module can focus on its prompt and contract.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Awaitable, Callable, TypeVar

from openai import APIError, APITimeoutError, AsyncAzureOpenAI, RateLimitError
from pydantic import BaseModel, ValidationError
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from llm_client import (
    FAST_MODEL,
    GENERATION_MODEL,
    REASONING_MODEL,
    RESTRICTED_PARAM_MODELS,
    get_fast_client,
    get_generation_client,
    get_reasoning_client,
)

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

    #: "fast" → gpt-4o-mini, "reasoning" → o4-mini, "generation" → gpt-5.4
    #: (see llm_client.py).
    tier: str = "reasoning"

    def __init__(self, *, agent_name: str, prompt_version: str) -> None:
        self.agent_name = agent_name
        self.prompt_version = prompt_version

    # ── Client + model name ──────────────────────────────────────────────
    def get_client(self) -> AsyncAzureOpenAI:
        if self.tier == "reasoning":
            client = get_reasoning_client()
        elif self.tier == "generation":
            client = get_generation_client()
        else:
            client = get_fast_client()
        if client is None:
            raise RuntimeError(
                "Azure OpenAI client unavailable — is AZURE_OPEN_AI_KEY set?"
            )
        return client

    @property
    def model_name(self) -> str:
        if self.tier == "reasoning":
            return REASONING_MODEL
        if self.tier == "generation":
            return GENERATION_MODEL
        return FAST_MODEL

    @property
    def _accepts_temperature(self) -> bool:
        """Standard chat models accept `temperature`; o-series and the gpt-5
        reasoning variants (e.g. gpt-5-mini) reject anything other than the
        default 1.0. Tier alone isn't enough to decide — gpt-5.4 (chat) and
        gpt-5-mini (reasoning) both run at tier="generation"."""
        if self.model_name in RESTRICTED_PARAM_MODELS:
            return False
        return self.tier in ("fast", "generation")

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
                        # pass them only for standard chat tiers.
                        if self._accepts_temperature:
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

    # ── Tool-use loop (function calling) ─────────────────────────────────
    async def call_with_tools(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        tools: list[dict],
        tool_dispatch: Callable[[str, dict], Awaitable[str]],
        terminator_tool: str,
        schema: type[T],
        max_iters: int = 6,
        temperature: float = 0.0,
    ) -> T:
        """Run an OpenAI function-calling loop until the model calls ``terminator_tool``.

        - ``tools`` is the OpenAI tool-schema list (each entry has ``type: "function"``,
          ``function: {name, description, parameters}``).
        - ``tool_dispatch(name, args)`` is called for every non-terminator tool; it
          must return a string that will be fed back as the ``role="tool"`` message.
        - When the model calls ``terminator_tool`` with its final payload, the
          payload is validated against ``schema`` and returned.
        - Iteration cap: after ``max_iters`` non-terminator tool calls, we force
          a final JSON-only completion (no tools) that must yield ``schema``.

        Transient OpenAI errors get 3 tenacity attempts with exponential backoff,
        same as :meth:`call_json`.
        """
        client = self.get_client()
        messages: list[dict] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        retry_policy = dict(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=1, max=8),
            retry=retry_if_exception_type(
                (RateLimitError, APITimeoutError, APIError)
            ),
            reraise=True,
        )

        for iteration in range(max_iters):
            kwargs: dict = {
                "model": self.model_name,
                "messages": messages,
                "tools": tools,
                "tool_choice": "auto",
            }
            if self._accepts_temperature:
                kwargs["temperature"] = temperature

            async for attempt in AsyncRetrying(**retry_policy):
                with attempt:
                    resp = await client.chat.completions.create(**kwargs)

            msg = resp.choices[0].message
            tool_calls = getattr(msg, "tool_calls", None) or []

            if not tool_calls:
                # Model replied with prose instead of calling a tool — nudge.
                content = msg.content or ""
                messages.append({"role": "assistant", "content": content})
                messages.append({
                    "role": "user",
                    "content": (
                        f"You did not call any tool. Call `{terminator_tool}` "
                        f"with your final answer to finish, or call one of the "
                        f"other tools to gather more signal first."
                    ),
                })
                continue

            # Attach the assistant message carrying the tool_calls to history.
            messages.append({
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in tool_calls
                ],
            })

            terminator_hit = False
            terminator_payload: dict | None = None

            for tc in tool_calls:
                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}

                if name == terminator_tool:
                    terminator_hit = True
                    terminator_payload = args
                    # Satisfy OpenAI's requirement that every tool_call has a
                    # matching tool message, even though we'll exit right after.
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "name": name,
                        "content": "ok",
                    })
                    continue

                try:
                    result = await tool_dispatch(name, args)
                except Exception as exc:  # noqa: BLE001
                    # Hard quota stops must propagate so the runner can save
                    # the checkpoint cleanly. Other tool errors are recoverable
                    # (bad URL, transient API hiccup) and become tool-message text.
                    from skill_library_v2.tools.exceptions import (
                        FirecrawlCreditsExhausted,
                    )
                    if isinstance(exc, FirecrawlCreditsExhausted):
                        raise
                    logger.warning(
                        "[%s] tool %s raised %s: %s",
                        self.agent_name, name, type(exc).__name__, exc,
                    )
                    result = f"error: {type(exc).__name__}: {exc}"

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": name,
                    "content": result if isinstance(result, str) else json.dumps(result),
                })

            if terminator_hit:
                try:
                    return schema.model_validate(terminator_payload or {})
                except ValidationError as exc:
                    # Corrective nudge — ask the model to re-emit valid args.
                    logger.warning(
                        "[%s] %s payload failed schema: %s",
                        self.agent_name, terminator_tool, exc,
                    )
                    messages.append({
                        "role": "user",
                        "content": (
                            f"Your `{terminator_tool}` arguments did not match "
                            f"the required schema. Error: {exc}. Call "
                            f"`{terminator_tool}` again with corrected arguments."
                        ),
                    })
                    continue

        # Hit the iteration cap — force a final no-tools JSON completion.
        logger.warning(
            "[%s] call_with_tools hit max_iters=%d; forcing JSON-only finalize.",
            self.agent_name, max_iters,
        )
        finalize_prompt = (
            f"You did not call `{terminator_tool}` in time. Emit the final "
            f"answer now as a JSON object matching the `{terminator_tool}` "
            f"arguments schema. No prose, no tool calls."
        )
        messages.append({"role": "user", "content": finalize_prompt})

        final_kwargs: dict = {
            "model": self.model_name,
            "messages": messages,
            "response_format": {"type": "json_object"},
        }
        if self._accepts_temperature:
            final_kwargs["temperature"] = temperature

        async for attempt in AsyncRetrying(**retry_policy):
            with attempt:
                resp = await client.chat.completions.create(**final_kwargs)
        raw = resp.choices[0].message.content or ""
        return parse_llm_json(raw, schema)

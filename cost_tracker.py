"""LLM API cost tracking for Azure OpenAI calls."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

# Azure OpenAI pricing (USD per 1M tokens)
_PRICING: dict[str, tuple[float, float]] = {
    "gpt-4.1-nano": (0.100, 0.400),
    "gpt-4.1-mini": (0.400, 1.600),
    "gpt-4.1":      (2.000, 8.000),
    "gpt-4o-mini":  (0.150, 0.600),
    "gpt-4o":       (2.500, 10.000),
    "gpt-5.4-mini": (0.750, 4.500),
    "gpt-5-mini":   (1.250, 5.000),
    "o4-mini":      (1.100, 4.400),
}
# Azure text-embedding-3-small — USD per 1M input tokens (no output tokens billed)
_EMBEDDING_PER_1M: dict[str, float] = {
    "text-embedding-3-small": 0.020,
    "text-embedding-3-large": 0.130,
}
_EMBEDDING_FALLBACK_PER_1M = 0.020
_FALLBACK_PRICING = (2.500, 10.000)


def _get_rates(model: str) -> tuple[float, float]:
    m = model.lower()
    for key, rates in _PRICING.items():
        if key in m:
            return rates
    return _FALLBACK_PRICING


@dataclass
class CostAccumulator:
    input_tokens: int = 0
    output_tokens: int = 0
    call_count: int = 0
    total_cost_usd: float = field(default=0.0)
    _by_model: dict = field(default_factory=dict)

    embedding_tokens: int = 0
    embedding_cost_usd: float = field(default=0.0)

    def add(self, model: str, input_tokens: int, output_tokens: int) -> None:
        input_rate, output_rate = _get_rates(model)
        cost = (input_tokens * input_rate + output_tokens * output_rate) / 1_000_000
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.call_count += 1
        self.total_cost_usd += cost
        entry = self._by_model.setdefault(model, {"input": 0, "output": 0, "cost": 0.0})
        entry["input"] += input_tokens
        entry["output"] += output_tokens
        entry["cost"] += cost

    def add_embedding(self, model: str, input_tokens: int) -> None:
        """Record Azure embedding API usage (input tokens only)."""
        if input_tokens <= 0:
            return
        m = model.lower()
        rate = _EMBEDDING_FALLBACK_PER_1M
        for key, r in _EMBEDDING_PER_1M.items():
            if key in m:
                rate = r
                break
        cost = (input_tokens * rate) / 1_000_000
        self.embedding_tokens += input_tokens
        self.embedding_cost_usd += cost
        self.total_cost_usd += cost
        self.call_count += 1
        key = f"{model} (embed)"
        entry = self._by_model.setdefault(key, {"input": 0, "output": 0, "cost": 0.0})
        entry["input"] += input_tokens
        entry["cost"] += cost

    def log_summary(self, endpoint: str, logger: logging.Logger) -> None:
        for model, m in self._by_model.items():
            logger.info(
                "[%s] %s — in=%d  out=%d  cost=$%.6f",
                endpoint, model, m["input"], m["output"], m["cost"],
            )
        if self.embedding_tokens:
            logger.info(
                "[%s] embeddings — tokens=%d  cost=$%.6f",
                endpoint,
                self.embedding_tokens,
                self.embedding_cost_usd,
            )
        logger.info(
            "[%s] TOTAL — calls=%d  in=%d  out=%d  cost=$%.6f",
            endpoint,
            self.call_count,
            self.input_tokens,
            self.output_tokens,
            self.total_cost_usd,
        )

"""LLM API cost tracking for Azure OpenAI calls."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

# Azure OpenAI pricing (USD per 1M tokens)
_PRICING: dict[str, tuple[float, float]] = {
    "gpt-5.4-mini": (0.750, 4.500),
    "gpt-4o-mini":  (0.150, 0.600),
    "o4-mini":      (1.100, 4.400),
}
_FALLBACK_PRICING = (0.750, 4.500)


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

    def add(self, model: str, input_tokens: int, output_tokens: int) -> None:
        input_rate, output_rate = _get_rates(model)
        cost = (input_tokens * input_rate + output_tokens * output_rate) / 1_000_000
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.call_count += 1
        self.total_cost_usd += cost

    def log_summary(self, endpoint: str, logger: logging.Logger) -> None:
        logger.info(
            "[%s] LLM cost — calls=%d  input_tokens=%d  output_tokens=%d  total_cost=$%.6f",
            endpoint,
            self.call_count,
            self.input_tokens,
            self.output_tokens,
            self.total_cost_usd,
        )

"""Dimension text embedder for Stage 3 sub-stage 3a.

Wraps an embed-text callable behind an in-memory cache so the same
(name, description) pair only hits the underlying API once per process.
The default embed_fn factory builds a closure over Azure's
``text-embedding-3-small`` deployment from :mod:`llm_client`; tests
inject a fake instead.
"""

from __future__ import annotations

from typing import Awaitable, Callable, Iterable


EmbedFn = Callable[[str], Awaitable[list[float]]]


def format_dim_text(name: str, description: str | None) -> str:
    """Canonical text input for embedding a dim. ``"<name>: <description>"``
    when both are non-empty; just the name when description is empty/None.
    Whitespace is trimmed at boundaries so cache keys stay stable across
    cosmetic LLM whitespace drift.
    """
    name_clean = (name or "").strip()
    desc_clean = (description or "").strip()
    if not desc_clean:
        return name_clean
    return f"{name_clean}: {desc_clean}"


class DimensionEmbedder:
    """In-memory cached embedder. ``embed_fn`` is injected so tests can
    fake out the network call. The cache is per-instance — pass the same
    embedder around within a single Stage 3 run to amortize repeats."""

    def __init__(self, *, embed_fn: EmbedFn) -> None:
        self._embed_fn = embed_fn
        self._cache: dict[str, list[float]] = {}

    async def embed(self, name: str, description: str | None) -> list[float]:
        text = format_dim_text(name, description)
        if text in self._cache:
            return self._cache[text]
        vec = await self._embed_fn(text)
        self._cache[text] = vec
        return vec

    async def embed_many(
        self, items: Iterable[tuple[str, str | None]]
    ) -> list[list[float]]:
        """Batch convenience. Repeated (name, description) pairs in the
        input list call ``embed_fn`` only once thanks to the cache; both
        output positions get the same cached vector."""
        out: list[list[float]] = []
        for name, desc in items:
            out.append(await self.embed(name, desc))
        return out


def make_default_embedder() -> DimensionEmbedder:
    """Glue: wire ``DimensionEmbedder`` to Azure's text-embedding-3-small
    deployment via :mod:`llm_client`. No unit test — this is a thin SDK
    adapter; behavior tests live in ``test_dim_embedder.py`` against the
    public :class:`DimensionEmbedder` interface."""
    from llm_client import EMBEDDING_MODEL, get_embedding_client

    async def _embed(text: str) -> list[float]:
        client = get_embedding_client()
        if client is None:
            raise RuntimeError(
                "Azure OpenAI embedding client unavailable — is "
                "AZURE_OPEN_AI_KEY set?"
            )
        resp = await client.embeddings.create(model=EMBEDDING_MODEL, input=text)
        return list(resp.data[0].embedding)

    return DimensionEmbedder(embed_fn=_embed)

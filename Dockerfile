# syntax=docker/dockerfile:1.7
# ──────────────────────────────────────────────────────────────────────────────
# Stage 1: Builder — compiles wheels, resolves deps, downloads spaCy models.
# ──────────────────────────────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Build-only toolchain. Stays in builder, never copied to runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ \
    && rm -rf /var/lib/apt/lists/*

# Resolve and install Python deps using the locked versions. This pulls
# torch==X.Y.Z+cu118 from the CUDA index per pyproject.toml's [tool.uv.sources].
# We keep the lockfile clean (don't strip CUDA there — that would break the
# GPU dev workflow) and replace torch with the CPU build at the end of the
# builder stage.
COPY pyproject.toml uv.lock .python-version ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project

# Project code + final install (so the project is registered in the venv).
COPY . .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen

# spaCy models. Install into the venv directly (uv pip is venv-aware here).
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --no-cache \
        https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl \
        https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-3.8.0/en_core_web_lg-3.8.0-py3-none-any.whl

# Replace CUDA torch with the CPU build. Must run AFTER the final
# `uv sync --frozen` above — otherwise that sync re-resolves against the
# lockfile and puts CUDA torch back. `uv pip install` is the pip-compat
# surface and honors --index-url (it does NOT read [tool.uv.sources], unlike
# `uv sync`/`uv add`/`uv lock`).
#
# We also explicitly uninstall the nvidia-* runtime libraries that came in
# with the CUDA wheel (~2.5 GB combined). uv doesn't auto-prune these when
# torch is reinstalled with a smaller dependency closure.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip uninstall \
        torch torchvision \
        nvidia-cublas-cu11 nvidia-cuda-cupti-cu11 nvidia-cuda-nvrtc-cu11 \
        nvidia-cuda-runtime-cu11 nvidia-cudnn-cu11 nvidia-cufft-cu11 \
        nvidia-curand-cu11 nvidia-cusolver-cu11 nvidia-cusparse-cu11 \
        nvidia-nccl-cu11 nvidia-nvtx-cu11 triton 2>/dev/null || true
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --no-cache \
        --index-url https://download.pytorch.org/whl/cpu \
        torch torchvision

# Strip pyc bytecode + tests bundled inside site-packages (huggingface, torch
# tests, etc.) — saves a few hundred MB without touching imports.
RUN find /app/.venv -depth \
        \( -type d -name "__pycache__" \
           -o -type d -name "tests" \
           -o -type d -name "test" \
        \) -exec rm -rf {} + 2>/dev/null || true
RUN find /app/.venv -name "*.pyc" -delete

# ──────────────────────────────────────────────────────────────────────────────
# Stage 2: Runtime — minimal image: python + venv + app code + runtime libs.
# ──────────────────────────────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

WORKDIR /app

# Runtime-only system libraries. docling/PIL need libgl, libxcb, etc.
# Critical: NO gcc/g++ here — those stayed in the builder.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libgomp1 \
    libglib2.0-0 \
    libxcb1 \
    libx11-6 \
    libxext6 \
    libxrender1 \
    libfontconfig1 \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

# Bring uv across so the entrypoint's `uv run` works.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy the resolved venv + app code from the builder. Single COPY each so the
# layer cache stays warm across small code changes.
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app /app

# Belt-and-suspenders: app code already does this in main.py too.
ENV CUDA_VISIBLE_DEVICES=""
# Skip rebuilding bytecode at runtime (we deleted it in builder for size).
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
# CRITICAL: prevent `uv run` from auto-syncing at container start. The venv
# was deliberately patched with CPU torch (different from the lockfile, which
# pins CUDA torch for the GPU dev workflow). Without this, uv would notice
# the drift and re-download 2.5 GB of nvidia-* + CUDA torch on first run,
# undoing the slimming.
ENV UV_NO_SYNC=1

EXPOSE 8000

CMD ["sh", "-c", "uv run uvicorn main:app --host 0.0.0.0 --port 8000 --workers ${UVICORN_WORKERS:-2}"]

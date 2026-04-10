FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# This service should stay CPU-only.
ENV CUDA_VISIBLE_DEVICES=""

# Docling's PDF preprocessing may require XCB/X11 libs even on CPU.
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ \
    # Docling/PDF preprocessing sometimes relies on OpenGL (libGL.so.1)
    libgl1 \
    # ONNX Runtime often needs OpenMP runtime
    libgomp1 \
    # Common runtime deps for GL/X11 stacks used by docling preprocess
    libglib2.0-0 \
    libxcb1 \
    libx11-6 \
    libxext6 \
    libxrender1 \
    libfontconfig1 \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files first for layer caching
COPY pyproject.toml uv.lock .python-version ./

RUN uv sync --frozen --no-install-project

# Copy application code
COPY . .

# Install the project itself
RUN uv sync --frozen

# Download spacy model
RUN uv pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl

EXPOSE 8000

CMD ["sh", "-c", "uv run uvicorn main:app --host 0.0.0.0 --port 8000 --workers ${UVICORN_WORKERS:-2}"]
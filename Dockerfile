FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/cache/huggingface \
    SCIKIT_LEARN_DATA=/cache/sklearn \
    BITSANDBYTES_NOWELCOME=1

# Install deps first (layer cache)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Copy application code
COPY backend/ backend/
COPY README.md .

RUN uv sync --frozen --no-dev

EXPOSE 8501 8000

WORKDIR /app/backend/user_interface

CMD ["uv", "run", "streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]

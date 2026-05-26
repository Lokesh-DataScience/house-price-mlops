FROM python:3.10-slim

WORKDIR /House-Price-MLOps

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# 1. Grab the official pre-compiled uv binary
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Install system runtime tools (Note: gcc/build-essential can often be removed if using uv)
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl libpq-dev \
 && rm -rf /var/lib/apt/lists/*

# 2. Copy dependencies early to leverage Docker cache
COPY requirements.txt .

# 3. Use uv to install packages directly into the system environment
RUN uv pip install --system --no-cache -r requirements.txt

# Copy app
COPY ./app ./app

# Create non-root user and fix permissions
RUN groupadd --system app && useradd --system --gid app appuser \
 && chown -R appuser:app /House-Price-MLOps

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
# ───────────────────── Stage 1: Build dependencies ─────────────────────
FROM python:3.9-slim AS build

WORKDIR /build

# Install build essentials only for the build stage
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        g++ \
        && rm -rf /var/lib/apt/lists/*

# Copy requirement file(s) first to leverage Docker layer caching
COPY requirements.txt .

# Install Python dependencies into /install (so we can copy them later)
RUN pip install --upgrade pip && \
    pip install --prefix=/install --no-cache-dir -r requirements.txt

# ───────────────────── Stage 2: Final runtime image ─────────────────────
FROM python:3.9-slim AS runtime

# Add a non‑root user for security
RUN useradd -m appuser

WORKDIR /app

# Copy installed site‑packages from the build stage
COPY --from=build /install /usr/local

# Copy application code *after* the deps to keep cache efficient
COPY . .

# Ensure expected directories exist (for uploads/models, etc.)
RUN mkdir -p /app/uploads /app/models /app/templates

# Change ownership (so the non‑root user can write to uploads/)
RUN chown -R appuser:appuser /app

# Expose Flask/Gunicorn port
EXPOSE 5000

USER appuser

# Gunicorn with 4 async workers (tweak if needed)
CMD ["gunicorn", "-k", "gevent", "--workers", "4", "--bind", "0.0.0.0:5000", "app:app"]

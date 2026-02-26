# BG-cleaner — CPU Docker image
# Model weights are mounted at runtime:
#   docker run -v ./assets/models:/app/assets/models:ro ...

FROM python:3.11-slim AS base

# Prevent Python from writing .pyc files and enable unbuffered output.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# System deps (none needed beyond what slim provides for PIL/numpy).
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------

COPY pyproject.toml ./

# Install the project dependencies (not the project itself yet).
RUN pip install --no-cache-dir pip --upgrade && \
    pip install --no-cache-dir .

# ---------------------------------------------------------------------------
# Application code
# ---------------------------------------------------------------------------

COPY src/ src/

# Re-install in editable mode so the package is importable.
RUN pip install --no-cache-dir -e .

# Create the models mount point.
RUN mkdir -p assets/models

# ---------------------------------------------------------------------------
# Streamlit configuration
# ---------------------------------------------------------------------------

# Disable Streamlit telemetry and browser auto-open.
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0

EXPOSE 8501

# Healthcheck: Streamlit exposes /_stcore/health.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

CMD ["streamlit", "run", "src/bgcleaner/ui/app.py"]
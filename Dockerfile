# --- Steg 1: Byggmiljö ---
# ✅ FIX: Uppgraderat till python:3.11-slim för att matcha kraven i pyproject.toml
FROM python:3.11-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src/ ./src/

# Bygg ett renodlat Python wheel-paket av din applikation
RUN pip install --no-cache-dir --upgrade pip \
    && pip wheel --no-cache-dir --no-deps --wheel-dir /app/wheels -e .

# --- Steg 2: Produktionsmiljö ---
# ✅ FIX: Samma här, synkat exekveringsmiljön till python:3.11-slim
FROM python:3.11-slim AS runner

WORKDIR /app

# Streamlit kräver port 8501 som standard
EXPOSE 8501

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0

COPY --from=builder /app/wheels /wheels
COPY --from=builder /app/pyproject.toml ./

# Installera applikationen från hjulet byggt i steg 1
RUN pip install --no-cache-dir /wheels/*

# Kopiera in de nödvändiga modulerna för att köra dashboarden och läsa data
COPY configs/ ./configs/
COPY data/ ./data/
COPY artifacts/ ./artifacts/
COPY dashboards/ ./dashboards/

# Startkommandot som kör igång webbappen på servern
CMD ["streamlit", "run", "dashboards/Home.py"]

# ─────────────────────────────────────────────────────────────────────────────
# Floyd-Warshall con Ray — Imagen Docker para reproducibilidad científica
#
# Imagen multi-etapa:
#   base  → Python + dependencias del sistema + LaTeX
#   deps  → dependencias Python (cacheables por layer)
#   final → imagen mínima de producción
#
# Requiere: nvidia-container-toolkit en el host para experimentos GPU.
# Sin GPU, los escenarios E_GPU* se omiten automáticamente.
# ─────────────────────────────────────────────────────────────────────────────

# ── Etapa base: sistema operativo y dependencias de sistema ─────────────────
FROM python:3.11-slim-bookworm AS base

LABEL org.opencontainers.image.title="Floyd-Warshall con Ray"
LABEL org.opencontainers.image.description="Entorno experimental reproducible para benchmarks de Floyd-Warshall paralelo con Ray"
LABEL org.opencontainers.image.source="https://github.com/mittods/Floyd-Warshall-Ray"
LABEL org.opencontainers.image.licenses="MIT"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    RAY_DISABLE_IMPORT_WARNING=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    procps \
    htop \
    sysstat \
    net-tools \
    iputils-ping \
    libfreetype6-dev \
    libpng-dev \
    pkg-config \
    texlive-latex-base \
    texlive-latex-extra \
    texlive-fonts-recommended \
    texlive-science \
    latexmk \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ── Etapa CPU: dependencias sin CUDA (test local / CI sin GPU) ──────────────
FROM base AS cpu

WORKDIR /app
COPY requirements-cpu.txt .
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements-cpu.txt

RUN mkdir -p resultados graficos metricas datasets logs \
    && chmod -R 777 resultados graficos metricas datasets logs
RUN groupadd -r investigador && useradd -r -m -g investigador investigador && \
    chown -R investigador:investigador /app

COPY --chown=investigador:investigador src/ ./src/
COPY --chown=investigador:investigador experimentos/ ./experimentos/
COPY --chown=investigador:investigador analisis/ ./analisis/
COPY --chown=investigador:investigador scripts/ ./scripts/
COPY --chown=investigador:investigador Makefile ./Makefile
COPY --chown=investigador:investigador docker/entrypoint.sh ./entrypoint.sh
RUN chmod +x scripts/*.sh entrypoint.sh

ENV FW_TAMANOS="256,512,1024,2048" \
    FW_WORKERS="1,2,4,8,16,32" \
    FW_REPETICIONES="5" \
    FW_SEMILLA="42" \
    FW_DENSIDAD="0.7" \
    FW_INTERVALO_MONITOR="0.5" \
    RAY_NUM_CPUS="" \
    RAY_OBJECT_STORE_MEMORY="" \
    CUPY_CACHE_DIR="/tmp/cupy_cache"

EXPOSE 8265
USER investigador
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["bash"]

# ── Etapa GPU: agrega CuPy + CUDA sobre el stage cpu ────────────────────────
# Build: docker build --target gpu -t floyd-warshall-ray:gpu .
FROM cpu AS gpu

USER root
RUN pip install "cupy-cuda12x[ctk]>=13.0.0" nvidia-ml-py>=12.0.0
USER investigador

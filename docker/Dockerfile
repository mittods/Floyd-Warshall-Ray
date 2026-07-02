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

# ── Etapa de dependencias Python ────────────────────────────────────────────
FROM base AS deps

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip setuptools wheel && \
    pip install numpy>=1.26.0 && \
    pip install -r requirements.txt

# ── Etapa final ──────────────────────────────────────────────────────────────
FROM deps AS final

WORKDIR /app

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

# Valores conservadores para uso local.
# En Patagon se sobreescriben con FW_TAMANOS="1024,2048,4096,8192" y FW_WORKERS="1,2,3".
ENV FW_TAMANOS="256,512,1024,2048" \
    FW_WORKERS="1,2,4,8,16,32" \
    FW_REPETICIONES="5" \
    FW_SEMILLA="42" \
    FW_DENSIDAD="0.7" \
    FW_INTERVALO_MONITOR="0.5" \
    RAY_NUM_CPUS="" \
    RAY_OBJECT_STORE_MEMORY="" \
    # Caché JIT de CuPy en disco local del contenedor (evita conflictos si
    # el mismo directorio de trabajo se monta en varios contenedores).
    CUPY_CACHE_DIR="/tmp/cupy_cache"

# Puerto del dashboard de Ray
EXPOSE 8265

USER investigador

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["bash"]

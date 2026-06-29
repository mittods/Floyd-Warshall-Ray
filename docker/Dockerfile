# ─────────────────────────────────────────────────────────────────────────────
# Floyd-Warshall con Ray — Imagen Docker para reproducibilidad científica
#
# Arquitectura de construcción multi-etapa:
#   etapa base  → Python + dependencias del sistema
#   etapa deps  → instalación de dependencias Python (cacheables)
#   etapa final → imagen mínima de producción
#
# Compatible con: AMD Ryzen Threadripper PRO 5975WX + NVIDIA TITAN V
# ─────────────────────────────────────────────────────────────────────────────

# ── Etapa base: sistema operativo y dependencias de sistema ─────────────────
FROM python:3.11-slim-bookworm AS base

LABEL org.opencontainers.image.title="Floyd-Warshall con Ray"
LABEL org.opencontainers.image.description="Entorno experimental reproducible para benchmarks de Floyd-Warshall paralelo con Ray"
LABEL org.opencontainers.image.source="https://github.com/martinmaza/floyd-warshall-ray"
LABEL org.opencontainers.image.licenses="MIT"

# Variables de entorno no interactivas
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    RAY_DISABLE_IMPORT_WARNING=1

# Instalar dependencias del sistema necesarias para:
#   - compilación de extensiones Python (build-essential)
#   - nvidia-smi para métricas GPU (nvidia-container-toolkit lo provee en runtime)
#   - herramientas de diagnóstico del sistema
#   - LaTeX para compilar el artículo (texlive-ieee)
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Herramientas de compilación
    build-essential \
    # Utilidades de sistema
    procps \
    htop \
    sysstat \
    # Herramientas de red y diagnóstico
    net-tools \
    iputils-ping \
    # Para generación de gráficos sin display
    libfreetype6-dev \
    libpng-dev \
    pkg-config \
    # LaTeX IEEE (para compilar el artículo)
    texlive-latex-base \
    texlive-latex-extra \
    texlive-fonts-recommended \
    texlive-science \
    latexmk \
    # Limpieza
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ── Etapa de dependencias Python ────────────────────────────────────────────
FROM base AS deps

WORKDIR /app

# Copiar solo requirements para aprovechar la caché de Docker
COPY requirements.txt .

# Instalar todas las dependencias Python
# Se separa numpy primero porque Ray depende de él
RUN pip install --upgrade pip setuptools wheel && \
    pip install numpy>=1.26.0 && \
    pip install -r requirements.txt

# ── Etapa final ──────────────────────────────────────────────────────────────
FROM deps AS final

WORKDIR /app

# Crear directorios de salida con permisos correctos
RUN mkdir -p \
    resultados \
    graficos \
    metricas \
    datasets \
    && chmod -R 777 resultados graficos metricas datasets

# Crear usuario no-root para mayor seguridad
RUN groupadd -r investigador && useradd -r -g investigador investigador && \
    chown -R investigador:investigador /app

# Copiar el código fuente del proyecto
COPY --chown=investigador:investigador src/ ./src/
COPY --chown=investigador:investigador experimentos/ ./experimentos/
COPY --chown=investigador:investigador analisis/ ./analisis/
COPY --chown=investigador:investigador scripts/ ./scripts/
COPY --chown=investigador:investigador docker/entrypoint.sh ./entrypoint.sh

# Hacer ejecutables los scripts
RUN chmod +x scripts/*.sh entrypoint.sh

# Variables de entorno para configuración del experimento
# Pueden sobreescribirse en docker-compose.yml o en la línea de comandos
ENV FW_TAMANOS="64,128,256,512,1024,2048" \
    FW_WORKERS="1,2,4,8,16,32" \
    FW_REPETICIONES="10" \
    FW_SEMILLA="42" \
    FW_DENSIDAD="0.7" \
    FW_INTERVALO_MONITOR="0.5" \
    RAY_NUM_CPUS="" \
    RAY_OBJECT_STORE_MEMORY=""

# Puerto del dashboard de Ray (opcional)
EXPOSE 8265

USER investigador

# El entrypoint permite ejecutar cualquier comando
# sin ejecutar nada automáticamente
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["bash"]

#!/bin/bash
# Comprime y exporta todos los resultados en un archivo tar.gz para archivar.
#
# Incluye: resultados/, graficos/, metricas/ y metadatos del experimento.

set -euo pipefail

PROYECTO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROYECTO_DIR"

COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "sin-git")
FECHA=$(date +"%Y%m%d_%H%M%S")
ARCHIVO="resultados_${FECHA}_${COMMIT}.tar.gz"

echo "Exportando resultados a: $ARCHIVO"

tar -czf "$ARCHIVO" \
    resultados/ \
    graficos/ \
    metricas/ \
    --exclude="*.pyc" \
    --exclude="__pycache__"

echo "Exportación completada: $ARCHIVO"
echo "Tamaño: $(du -sh "$ARCHIVO" | cut -f1)"

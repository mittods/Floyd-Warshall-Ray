#!/bin/bash
# Limpia los resultados, gráficos y métricas generados por los benchmarks.
# Solicita confirmación antes de eliminar.

set -euo pipefail

PROYECTO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROYECTO_DIR"

echo "Este script eliminará el contenido de:"
echo "  - resultados/"
echo "  - graficos/"
echo "  - metricas/"
echo ""
read -r -p "¿Confirmar limpieza? [s/N]: " CONFIRM

if [[ "$CONFIRM" =~ ^[sS]$ ]]; then
    find resultados/ -type f \( -name "*.json" -o -name "*.csv" -o -name "*.parquet" -o -name "*.tex" \) -delete
    find graficos/ -type f \( -name "*.pdf" -o -name "*.png" \) -delete
    find metricas/ -type f -delete
    echo "Limpieza completada."
else
    echo "Limpieza cancelada."
fi

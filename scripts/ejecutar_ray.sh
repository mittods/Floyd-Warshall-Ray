#!/bin/bash
# Ejecuta únicamente los benchmarks de la versión paralela con Ray.
#
# Uso: ./scripts/ejecutar_ray.sh [--workers 4,8,16,32] [--n 512,1024]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(dirname "$SCRIPT_DIR")"

export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

echo "Ejecutando benchmarks paralelos con Ray..."
echo "Configuración:"
echo "  Tamaños: ${FW_TAMANOS:-64,128,256,512,1024,2048}"
echo "  Workers: ${FW_WORKERS:-1,2,4,8,16,32}"
echo "  Repeticiones: ${FW_REPETICIONES:-10}"
echo ""

python -m experimentos.ejecutar_benchmarks \
    --escenarios E1_comparacion E2_escal_fuerte E3_escal_debil E4_overhead E5_carga_maxima \
    "$@"

echo ""
echo "Benchmarks Ray completados."
echo "Resultados en: resultados/"

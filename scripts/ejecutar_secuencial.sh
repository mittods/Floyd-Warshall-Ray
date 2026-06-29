#!/bin/bash
# Ejecuta únicamente los benchmarks de la versión secuencial.
#
# Uso: ./scripts/ejecutar_secuencial.sh [--n 256,512,1024] [--repeticiones 10]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(dirname "$SCRIPT_DIR")"

export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

echo "Ejecutando benchmarks secuenciales..."
echo "Configuración:"
echo "  Tamaños: ${FW_TAMANOS:-64,128,256,512,1024,2048}"
echo "  Repeticiones: ${FW_REPETICIONES:-10}"
echo ""

python -m experimentos.ejecutar_benchmarks \
    --escenarios E1_comparacion E5_carga_maxima \
    "$@"

echo ""
echo "Benchmarks secuenciales completados."
echo "Resultados en: resultados/"

#!/bin/bash
# Genera los gráficos a partir de los resultados almacenados.
set -euo pipefail
cd "$(dirname "$(dirname "${BASH_SOURCE[0]}")")"
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
echo "Generando gráficos..."
python -m analisis.generar_graficos
echo "Gráficos generados en: graficos/"

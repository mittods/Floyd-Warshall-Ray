#!/bin/bash
# Genera las tablas LaTeX a partir de los resultados almacenados.
set -euo pipefail
cd "$(dirname "$(dirname "${BASH_SOURCE[0]}")")"
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
echo "Generando tablas LaTeX..."
python -m analisis.generar_tablas
echo "Tablas generadas en: resultados/*.tex"

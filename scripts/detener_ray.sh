#!/bin/bash
# Detiene el cluster Ray local.

set -euo pipefail

echo "Deteniendo Ray..."

if ! command -v ray &>/dev/null; then
    echo "Error: Ray no está instalado o no está en el PATH."
    exit 1
fi

ray stop --force

echo "Ray detenido."

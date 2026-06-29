#!/bin/bash
# Valida que el entorno tenga todas las dependencias necesarias
# para ejecutar los experimentos correctamente.
#
# Verifica: Python, Ray, librerías, permisos de directorios, GPU.

set -euo pipefail

PROYECTO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROYECTO_DIR"

export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

ERRORES=0

check() {
    local NOMBRE="$1"
    local CMD="$2"
    if eval "$CMD" &>/dev/null; then
        echo "  [OK] $NOMBRE"
    else
        echo "  [FALLO] $NOMBRE"
        ERRORES=$((ERRORES + 1))
    fi
}

echo "══════════════════════════════════════════════════"
echo "  Validación del entorno experimental"
echo "══════════════════════════════════════════════════"
echo ""

echo "── Python y dependencias ─────────────────────────"
check "Python 3.10+" "python -c 'import sys; assert sys.version_info >= (3,10)'"
check "Ray" "python -c 'import ray'"
check "NumPy" "python -c 'import numpy'"
check "SciPy" "python -c 'import scipy'"
check "Pandas" "python -c 'import pandas'"
check "Matplotlib" "python -c 'import matplotlib'"
check "psutil" "python -c 'import psutil'"
check "pynvml" "python -c 'import pynvml'"
check "pyarrow" "python -c 'import pyarrow'"
echo ""

echo "── Módulos del proyecto ──────────────────────────"
check "Módulo secuencial" "python -c 'from src.secuencial import floyd_warshall_secuencial'"
check "Módulo Ray" "python -c 'from src.ray_parallel import floyd_warshall_ray'"
check "Módulo monitor" "python -c 'from src.utils import MonitorSistema'"
check "Módulo exportador" "python -c 'from src.utils import exportar_resultados'"
check "Módulo experimentos" "python -c 'from experimentos import generar_escenarios'"
echo ""

echo "── Directorios de salida ─────────────────────────"
for dir in resultados graficos metricas datasets; do
    if [ -d "$dir" ] && [ -w "$dir" ]; then
        echo "  [OK] $dir/ (escritura habilitada)"
    else
        mkdir -p "$dir"
        echo "  [CREADO] $dir/"
    fi
done
echo ""

echo "── GPU ────────────────────────────────────────────"
if command -v nvidia-smi &>/dev/null; then
    GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    echo "  [OK] GPU detectada: $GPU"
    nvidia-smi --query-gpu=memory.total,driver_version --format=csv,noheader
else
    echo "  [AVISO] nvidia-smi no disponible (GPU no será monitoreada)"
fi
echo ""

echo "── Prueba funcional mínima ───────────────────────"
check "Floyd-Warshall secuencial n=64" "python -c \"
import sys; sys.path.insert(0, '.')
from src.secuencial.floyd_warshall_secuencial import floyd_warshall_secuencial, inicializar_matriz
import numpy as np
m = inicializar_matriz(64, semilla=0)
resultado, _ = floyd_warshall_secuencial(m)
assert resultado.shape == (64, 64)
assert np.all(np.diag(resultado) == 0)
\""

check "Ray inicializable" "python -c \"
import ray
ray.init(ignore_reinit_error=True, num_cpus=2)
assert ray.is_initialized()
ray.shutdown()
\""
echo ""

echo "══════════════════════════════════════════════════"
if [ "$ERRORES" -eq 0 ]; then
    echo "  Validación exitosa. Entorno listo."
else
    echo "  ERRORES: $ERRORES. Revisar dependencias faltantes."
    exit 1
fi
echo "══════════════════════════════════════════════════"

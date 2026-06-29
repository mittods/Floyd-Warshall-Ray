#!/bin/bash
# Inicia el cluster Ray local en el sistema host.
# Útil para ejecutar los benchmarks fuera de Docker.
#
# Uso: ./scripts/iniciar_ray.sh [--num-cpus N] [--num-gpus N]

set -euo pipefail

NUM_CPUS="${RAY_NUM_CPUS:-$(nproc)}"
NUM_GPUS="${RAY_NUM_GPUS:-0}"
PUERTO_DASHBOARD="${RAY_DASHBOARD_PORT:-8265}"

# Verificar si Ray ya está corriendo
if ray status &>/dev/null 2>&1; then
    echo "Ray ya está activo. Usando el cluster existente."
    ray status
    exit 0
fi

echo "Iniciando Ray..."
echo "  CPUs: $NUM_CPUS"
echo "  GPUs: $NUM_GPUS"
echo "  Dashboard: http://localhost:$PUERTO_DASHBOARD"
echo ""

ray start \
    --head \
    --num-cpus "$NUM_CPUS" \
    --num-gpus "$NUM_GPUS" \
    --dashboard-host 0.0.0.0 \
    --dashboard-port "$PUERTO_DASHBOARD" \
    --disable-usage-stats

echo ""
echo "Ray iniciado correctamente."
echo "Dashboard: http://localhost:$PUERTO_DASHBOARD"
echo ""
echo "Para detener Ray: ./scripts/detener_ray.sh"

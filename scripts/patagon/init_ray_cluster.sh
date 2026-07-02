#!/bin/bash
# Inicializa un cluster Ray multi-nodo sobre Slurm.
#
# Uso desde un sbatch multi-nodo:
#   --nodes=N --ntasks-per-node=1
#   srun bash scripts/patagon/init_ray_cluster.sh
#
# Variables de entorno requeridas (Slurm las inyecta automáticamente):
#   SLURM_NODEID        — 0 en el head node, 1..N-1 en workers
#   SLURM_NODELIST      — lista de nodos asignados
#   SLURM_CPUS_PER_TASK — CPUs por tarea
#   CUDA_VISIBLE_DEVICES — set por Slurm cuando se pide --gres=gpu:N

set -euo pipefail

RAY_PORT=6380
RAY_DASHBOARD_PORT=8265
HEAD_NODE=$(scontrol show hostnames "$SLURM_NODELIST" | head -1)
WORKER_NODES=$(scontrol show hostnames "$SLURM_NODELIST" | tail -n +2)

NUM_CPUS="${SLURM_CPUS_PER_TASK:-1}"
NUM_GPUS=$(echo "${CUDA_VISIBLE_DEVICES:-}" | tr ',' '\n' | grep -c '[0-9]' || echo 0)

if [[ "$SLURM_NODEID" -eq 0 ]]; then
    echo "[Ray] Iniciando head node en $HEAD_NODE:$RAY_PORT"
    ray start \
        --head \
        --port="$RAY_PORT" \
        --dashboard-port="$RAY_DASHBOARD_PORT" \
        --num-cpus="$NUM_CPUS" \
        --num-gpus="$NUM_GPUS" \
        --block &
    HEAD_PID=$!

    # Esperar a que el head esté listo
    sleep 10

    echo "[Ray] Head node activo. Dashboard: http://${HEAD_NODE}:${RAY_DASHBOARD_PORT}"
    echo "RAY_ADDRESS=${HEAD_NODE}:${RAY_PORT}" > /tmp/ray_address.env
    wait "$HEAD_PID"
else
    # Worker: esperar a que el head esté listo antes de conectar
    sleep 15
    echo "[Ray] Conectando worker $SLURM_NODEID → $HEAD_NODE:$RAY_PORT"
    ray start \
        --address="${HEAD_NODE}:${RAY_PORT}" \
        --num-cpus="$NUM_CPUS" \
        --num-gpus="$NUM_GPUS" \
        --block
fi

#!/bin/bash
#SBATCH --job-name=fw-ray-gpu
#SBATCH --partition=ai
#SBATCH --gpus=8
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=logs/fw_gpu_%j.out
#SBATCH --error=logs/fw_gpu_%j.err

mkdir -p logs

# Grupos GPU: baseline 1 GPU (A100) + multi-GPU (hasta 8 A100).
# Ray detecta automáticamente las 8 GPUs via CUDA_VISIBLE_DEVICES inyectado por Slurm.
# No se requiere ray start externo: ray.init() sin argumentos inicializa en el nodo local.

srun \
    --container-name=fw-ray \
    --container-workdir="${PWD}" \
    python -m experimentos.ejecutar_benchmarks \
        --escenarios E_GPU E_GPU_blocked E_GPU_multi E_GPU_blocked_multi

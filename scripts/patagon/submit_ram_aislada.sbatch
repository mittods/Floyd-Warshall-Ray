#!/bin/bash
#SBATCH --job-name=fw-ram-aislada
#SBATCH --partition=A4000
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/fw_ram_aislada_%j.out
#SBATCH --error=logs/fw_ram_aislada_%j.err

mkdir -p logs

# Solo se necesita 1 GPU: las 3 variantes medidas aquí (gpu_secuencial,
# gpu_ray, gpu_blocked) son de un único dispositivo.
#
# Cada combinación (algoritmo, n, repetición) se ejecuta en un proceso
# nuevo (ver experimentos/medir_ram_aislado.py), así que el caché JIT de
# CuPy en disco evita recompilar el RawKernel de gpu_blocked en cada uno.
export CUPY_CACHE_DIR=/tmp/cupy_cache_${SLURM_JOB_ID}

srun python3 -m experimentos.medir_ram_aislado

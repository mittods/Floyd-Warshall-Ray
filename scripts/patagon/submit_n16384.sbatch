#!/bin/bash
#SBATCH --job-name=fw-ray-gpu-16384
#SBATCH --partition=A4000
#SBATCH --gpus=3
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=logs/fw_gpu_16384_%j.out
#SBATCH --error=logs/fw_gpu_16384_%j.err

mkdir -p logs

export FW_TAMANOS="1024,2048,4096,8192,16384"
export FW_WORKERS="1,2,3"
export FW_REPETICIONES=3
export CUPY_CACHE_DIR=/tmp/cupy_cache_${SLURM_JOB_ID}

srun python3 -m experimentos.ejecutar_benchmarks \
    --escenarios E_GPU E_GPU_blocked E_GPU_multi E_GPU_blocked_multi \
    --n 16384 \
    --precargar

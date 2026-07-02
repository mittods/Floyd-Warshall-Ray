#!/bin/bash
#SBATCH --job-name=fw-ray-gpu
#SBATCH --partition=AI
#SBATCH --gpus=8
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=logs/fw_gpu_%j.out
#SBATCH --error=logs/fw_gpu_%j.err

mkdir -p logs

# Ray detecta las 8 GPUs automáticamente via CUDA_VISIBLE_DEVICES de Slurm
srun python3 -m experimentos.ejecutar_benchmarks \
    --escenarios E_GPU E_GPU_blocked E_GPU_multi E_GPU_blocked_multi

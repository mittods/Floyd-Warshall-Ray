#!/bin/bash
#SBATCH --job-name=fw-ray-gpu
#SBATCH --partition=A4000
#SBATCH --gpus=3
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/fw_gpu_%j.out
#SBATCH --error=logs/fw_gpu_%j.err

mkdir -p logs

# A4000: 3x RTX A4000 16GB. Máximo 3 workers GPU.
# n=8192 → matriz float64 = 512MB, entra bien en 16GB de VRAM.
export FW_TAMANOS="1024,2048,4096,8192"
export FW_WORKERS="1,2,3"
export FW_REPETICIONES=5

# Ray detecta las 8 GPUs automáticamente via CUDA_VISIBLE_DEVICES de Slurm
srun python3 -m experimentos.ejecutar_benchmarks \
    --escenarios E_GPU E_GPU_blocked E_GPU_multi E_GPU_blocked_multi

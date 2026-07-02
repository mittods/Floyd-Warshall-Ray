#!/bin/bash
#SBATCH --job-name=fw-ray-cpu
#SBATCH --partition=AI
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=logs/fw_cpu_%j.out
#SBATCH --error=logs/fw_cpu_%j.err

mkdir -p logs

export FW_TAMANOS="128,256,512,1024,2048,4096,8192"
export FW_WORKERS="1,2,4,8,16,32,64"
export FW_REPETICIONES=10
export OMP_NUM_THREADS=1

srun python3 -m experimentos.ejecutar_benchmarks \
    --escenarios E1_comparacion E2_escal_fuerte E3_escal_debil E4_overhead E5_carga_maxima

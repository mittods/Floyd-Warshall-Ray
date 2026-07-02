#!/bin/bash
#SBATCH --job-name=fw-cpu-8192
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/fw_cpu_8192_%j.out
#SBATCH --error=logs/fw_cpu_8192_%j.err

mkdir -p logs

export FW_TAMANOS="128,256,512,1024,2048,4096,8192"
export FW_WORKERS="1,2,4,8,16,32"
export FW_REPETICIONES=3
export OMP_NUM_THREADS=1

srun python3 -m experimentos.ejecutar_benchmarks \
    --escenarios E1_comparacion E5_carga_maxima \
    --n 8192 \
    --precargar

"""
Configuraciones predefinidas para el cluster Patagon (UACh).

Particiones disponibles:
    ai     — 8× NVIDIA A100 40GB, nodo DGX
    cpu    — 64× AMD EPYC 9534 (Genoa, 2.45 GHz), 256 GB RAM

Uso típico:
    from experimentos.config_patagon import PATAGON_GPU, PATAGON_CPU
    escenarios = generar_escenarios(PATAGON_GPU)

O mediante variables de entorno en el sbatch:
    export FW_TAMANOS="1024,2048,4096,8192,16384"
    export FW_WORKERS="1,2,4,8"
    export FW_REPETICIONES=5
"""
from .config import ConfigExperimento

# ── Partición AI: 8× A100 40GB ──────────────────────────────────────────────
# Tamaños empezando en 1024 (n=512 es demasiado pequeño para amortizar IPC).
# n=16384 → matriz float64: 16384² × 8B ≈ 2 GB por actor con 8 GPUs.
# workers=[1,2,4,8]: un actor por GPU física.
PATAGON_GPU = ConfigExperimento(
    tamanos_matriz=[1024, 2048, 4096, 8192, 16384],
    workers_ray=[1, 2, 4, 8],
    num_repeticiones=5,
    densidad_grafo=0.3,
    semilla_base=42,
    nivel_confianza=0.95,
    intervalo_monitor_s=0.1,
)

# ── Partición CPU: 64× EPYC 9534 ────────────────────────────────────────────
# EPYC 9534 tiene 64 cores físicos con SMT → 128 threads.
# workers=[1,2,4,8,16,32,64]: un actor por bloque de cores.
# Con 64 workers, cada actor tiene 1 core dedicado (muy bajo overhead de OS).
# Tamaños más amplios que local: la memoria RAM (256 GB) permite n=16384
# (matriz float64: 16384² × 8B ≈ 2 GB; 64 actores × 2 GB = manejable).
PATAGON_CPU = ConfigExperimento(
    tamanos_matriz=[128, 256, 512, 1024, 2048, 4096, 8192, 16384],
    workers_ray=[1, 2, 4, 8, 16, 32, 64],
    num_repeticiones=10,
    densidad_grafo=0.3,
    semilla_base=42,
    nivel_confianza=0.95,
    intervalo_monitor_s=0.5,
)

# ── Configuración reducida para pruebas rápidas en Patagon (smoke test) ──────
PATAGON_SMOKE = ConfigExperimento(
    tamanos_matriz=[512, 1024, 2048],
    workers_ray=[1, 2, 4, 8],
    num_repeticiones=2,
    densidad_grafo=0.3,
    semilla_base=42,
    nivel_confianza=0.95,
    intervalo_monitor_s=0.1,
)

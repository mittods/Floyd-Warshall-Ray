"""
Implementación secuencial de Floyd-Warshall sobre GPU mediante CuPy.

CuPy reemplaza NumPy con kernels CUDA: cada operación vectorizada se ejecuta
en la GPU usando miles de hilos CUDA en paralelo. El bucle externo sobre k
sigue siendo secuencial (misma restricción de dependencias que en CPU), pero
cada actualización de la matriz completa se ejecuta en la GPU.

Flujo de datos:
    CPU (numpy) → GPU (cupy) → n iteraciones k en GPU → CPU (numpy)

La transferencia inicial y final se mide por separado del cómputo puro para
permitir reportar:
    - t_transferencia: tiempo de copia CPU→GPU + GPU→CPU
    - t_calculo: tiempo de las n iteraciones de Floyd-Warshall en GPU
    - t_total: suma de ambos (lo que experimenta el usuario)

Nota sobre sincronización:
    Las operaciones CuPy son asíncronas por defecto. Se usa
    cp.cuda.Stream.null.synchronize() al medir tiempos para garantizar
    que el kernel terminó antes de registrar el tiempo.
"""
import logging
import time
from typing import Tuple

import numpy as np

logger = logging.getLogger(__name__)


def gpu_disponible() -> bool:
    """Retorna True si CuPy puede acceder a una GPU CUDA."""
    try:
        import cupy as cp
        cp.array([1])  # operación mínima para verificar acceso real
        return True
    except Exception:
        return False


def floyd_warshall_gpu(
    distancias: np.ndarray,
) -> Tuple[np.ndarray, dict]:
    """
    Ejecuta Floyd-Warshall secuencial sobre GPU con CuPy.

    Args:
        distancias: Matriz de adyacencia float64 de tamaño n×n (en CPU).

    Returns:
        Tupla (matriz_resultado_cpu, métricas_dict).
    """
    try:
        import cupy as cp
    except ImportError:
        raise RuntimeError(
            "CuPy no está instalado. Instalar con: pip install cupy-cuda12x"
        )

    n = distancias.shape[0]

    # ── Transferencia CPU → GPU ──────────────────────────────────────────────
    t_trans_inicio = time.perf_counter()
    dist_gpu = cp.asarray(distancias, dtype=cp.float64)
    cp.cuda.Stream.null.synchronize()
    t_transferencia_ida = time.perf_counter() - t_trans_inicio

    # ── Cómputo en GPU ───────────────────────────────────────────────────────
    tiempos_por_k: list = []
    t_calculo_inicio = time.perf_counter()

    for k in range(n):
        t_k = time.perf_counter()

        # Broadcasting vectorizado: actualización completa de la matriz en GPU
        nueva_via = dist_gpu[:, k:k+1] + dist_gpu[k:k+1, :]
        cp.minimum(dist_gpu, nueva_via, out=dist_gpu)

        # Sincronizar para medición precisa (mínimo overhead)
        cp.cuda.Stream.null.synchronize()
        tiempos_por_k.append(time.perf_counter() - t_k)

    cp.cuda.Stream.null.synchronize()
    t_calculo = time.perf_counter() - t_calculo_inicio

    # ── Transferencia GPU → CPU ──────────────────────────────────────────────
    t_vuelta_inicio = time.perf_counter()
    resultado = cp.asnumpy(dist_gpu)
    cp.cuda.Stream.null.synchronize()
    t_transferencia_vuelta = time.perf_counter() - t_vuelta_inicio

    t_transferencia = t_transferencia_ida + t_transferencia_vuelta
    t_total = t_transferencia + t_calculo

    tiempos_arr = np.array(tiempos_por_k)
    metricas = {
        "algoritmo": "gpu_secuencial",
        "n": n,
        "num_actores": 0,
        "tiempo_total_s": float(t_total),
        "tiempo_calculo_s": float(t_calculo),
        "tiempo_transferencia_s": float(t_transferencia),
        "tiempo_transferencia_ida_s": float(t_transferencia_ida),
        "tiempo_transferencia_vuelta_s": float(t_transferencia_vuelta),
        "overhead_ray_s": 0.0,
        "tiempo_promedio_k_s": float(tiempos_arr.mean()),
        "tiempo_std_k_s": float(tiempos_arr.std()),
        "tiempo_min_k_s": float(tiempos_arr.min()),
        "tiempo_max_k_s": float(tiempos_arr.max()),
        "tiempos_por_k": tiempos_por_k,
    }

    logger.info(
        "Floyd-Warshall GPU secuencial: n=%d, cómputo=%.4fs, transferencia=%.4fs",
        n, t_calculo, t_transferencia,
    )
    return resultado, metricas

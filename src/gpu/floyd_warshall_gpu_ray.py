"""
Implementación de Floyd-Warshall con CuPy sobre GPU coordinada por Ray.

Diseño con actor único:
    Con una sola GPU (NVIDIA TITAN V), no tiene sentido particionar la
    matriz entre múltiples actores GPU — todos competirían por el mismo
    dispositivo y la transferencia entre actores pasaría por CPU.

    En cambio, se usa un único actor Ray que:
      1. Mantiene la matriz completa en memoria GPU (cupy array).
      2. Recibe solo el índice k por iteración (sin transferencia de datos).
      3. Ejecuta la actualización completa de la matriz en GPU.
      4. Al finalizar, retorna la matriz a CPU.

    Esto permite medir con precisión el overhead de Ray (scheduling,
    llamadas .remote(), ray.get()) sobre una GPU que ya es internamente
    paralela, respondiendo la pregunta: ¿añade Ray valor cuando el
    cómputo ya es masivamente paralelo?

Overhead esperado:
    Cada iteración k genera 1 llamada .remote() y 1 ray.get(), con
    latencia del scheduler de ~0.1-1ms. Para n=1024: ~1024 ms de
    overhead puro de Ray, independiente del tamaño de la GPU.
    Este overhead es el mismo que en la versión CPU Ray, pero relativo
    al tiempo de cómputo GPU (mucho más rápido) es proporcionalmente
    mayor, lo cual es el hallazgo central a medir.
"""
import logging
import time
from typing import Tuple

import numpy as np
import ray

logger = logging.getLogger(__name__)


@ray.remote(num_gpus=1)
class GPUActor:
    """
    Actor Ray que mantiene la matriz de distancias en GPU y la actualiza.

    Se declara num_gpus=1 para que Ray reserve la GPU exclusivamente
    para este actor durante toda su vida útil.
    """

    def __init__(self, distancias: np.ndarray):
        import cupy as cp
        self.cp = cp
        self.dist = cp.asarray(distancias, dtype=cp.float64)
        self.n = distancias.shape[0]

    def actualizar_k(self, k: int) -> None:
        """Ejecuta la iteración k de Floyd-Warshall en GPU."""
        nueva_via = self.dist[:, k:k+1] + self.dist[k:k+1, :]
        self.cp.minimum(self.dist, nueva_via, out=self.dist)
        self.cp.cuda.Stream.null.synchronize()

    def obtener_resultado(self) -> np.ndarray:
        """Transfiere la matriz final de GPU a CPU y la retorna."""
        resultado = self.cp.asnumpy(self.dist)
        return resultado


def floyd_warshall_gpu_ray(
    distancias: np.ndarray,
    inicializar: bool = True,
) -> Tuple[np.ndarray, dict]:
    """
    Ejecuta Floyd-Warshall en GPU con el bucle k coordinado por Ray.

    Args:
        distancias: Matriz de adyacencia float64 n×n (en CPU).
        inicializar: Si True, inicializa Ray si no está activo.

    Returns:
        Tupla (matriz_resultado_cpu, métricas_dict).
    """
    try:
        import cupy as cp  # noqa: F401 — verifica disponibilidad
    except ImportError:
        raise RuntimeError(
            "CuPy no está instalado. Instalar con: pip install cupy-cuda12x"
        )

    if inicializar and not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    n = distancias.shape[0]

    # Verificar que Ray tiene GPU disponible
    gpus_disponibles = ray.cluster_resources().get("GPU", 0)
    if gpus_disponibles < 1:
        raise RuntimeError(
            "Ray no detecta GPU. Asegúrese de que el contenedor tiene acceso "
            "a GPU (runtime: nvidia) y que Ray se inició con num_gpus≥1."
        )

    # ── Crear actor y transferir matriz a GPU ────────────────────────────────
    t_setup_inicio = time.perf_counter()
    actor = GPUActor.remote(distancias)
    # Esperar a que el actor esté listo (la transferencia CPU→GPU ocurre aquí)
    ray.get(actor.actualizar_k.remote(0))  # warm-up implícito en k=0
    t_setup = time.perf_counter() - t_setup_inicio

    # ── Bucle k coordinado por Ray ───────────────────────────────────────────
    tiempos_por_k: list = []
    t_calculo_inicio = time.perf_counter()

    # k=0 ya ejecutado en el setup; reejecutar limpiamente desde 0
    # Recrear el actor para medición justa
    actor = GPUActor.remote(distancias)

    t_calculo_inicio = time.perf_counter()
    for k in range(n):
        t_k = time.perf_counter()
        ray.get(actor.actualizar_k.remote(k))
        tiempos_por_k.append(time.perf_counter() - t_k)

    t_calculo = time.perf_counter() - t_calculo_inicio

    # ── Recuperar resultado ──────────────────────────────────────────────────
    t_vuelta_inicio = time.perf_counter()
    resultado = ray.get(actor.obtener_resultado.remote())
    t_vuelta = time.perf_counter() - t_vuelta_inicio

    t_total = t_setup + t_calculo + t_vuelta

    tiempos_arr = np.array(tiempos_por_k)
    metricas = {
        "algoritmo": "gpu_ray",
        "n": n,
        "num_actores": 1,
        "tiempo_total_s": float(t_total),
        "tiempo_calculo_s": float(t_calculo),
        "tiempo_setup_s": float(t_setup),
        "tiempo_transferencia_s": float(t_vuelta),
        "overhead_ray_s": float(t_setup + t_vuelta),
        "tiempo_promedio_k_s": float(tiempos_arr.mean()),
        "tiempo_std_k_s": float(tiempos_arr.std()),
        "tiempo_min_k_s": float(tiempos_arr.min()),
        "tiempo_max_k_s": float(tiempos_arr.max()),
        "tiempos_por_k": tiempos_por_k,
    }

    logger.info(
        "Floyd-Warshall GPU+Ray: n=%d, cómputo=%.4fs, overhead=%.4fs",
        n, t_calculo, t_setup + t_vuelta,
    )
    return resultado, metricas

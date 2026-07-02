"""
Floyd-Warshall naïve multi-GPU coordinado por Ray.

Diseño: p actores Ray, cada uno con num_gpus=1 (una A100 dedicada).
Cada actor mantiene su partición de filas en la VRAM de su GPU.

Por iteración k:
  1. El actor dueño de la fila k la transfiere GPU→CPU→object store.
  2. Todos los actores reciben row_k via object store (CPU→GPU de cada uno).
  3. Cada actor actualiza sus filas en paralelo sobre su GPU.
  4. Barrera: ray.get(futures).

La comunicación es O(n) por iteración (solo row_k), igual que ray_actores
CPU, pero el cómputo dentro de cada actor usa miles de CUDA cores en lugar
de threads del sistema operativo. Sobre 8 A100 esto es paralelismo real:
cada GPU ejecuta genuinamente en paralelo con las demás.
"""
import logging
import time
from typing import Tuple

import numpy as np
import ray

logger = logging.getLogger(__name__)


@ray.remote(num_gpus=1)
class GPUFilasActor:
    """
    Actor Ray con partición de filas almacenada en VRAM propia.

    Ray asigna una GPU física distinta a cada instancia mediante
    CUDA_VISIBLE_DEVICES, por lo que cp.asarray opera sobre la
    GPU local del actor sin conflicto con otros actores.
    """

    def __init__(self, bloque: np.ndarray, inicio_fila: int):
        import os
        # Caché JIT por proceso: evita deadlock cuando múltiples actores
        # compilan kernels CUDA simultáneamente en el mismo directorio.
        os.environ['CUPY_CACHE_DIR'] = f'/tmp/cupy_cache_{os.getpid()}'
        import cupy as cp
        self.cp = cp
        self.dist = cp.asarray(bloque, dtype=cp.float64)
        self.inicio = inicio_fila
        self.fin = inicio_fila + bloque.shape[0]
        self.n = bloque.shape[1]

    def obtener_fila(self, k: int):
        """Retorna la fila global k como numpy array (GPU → CPU)."""
        idx_local = k - self.inicio
        if 0 <= idx_local < self.dist.shape[0]:
            return self.cp.asnumpy(self.dist[idx_local, :])
        return None

    def actualizar_bloque(self, k: int, row_k: np.ndarray) -> None:
        """Actualiza la partición local para la iteración k (CuPy vectorizado)."""
        row_k_gpu = self.cp.asarray(row_k)
        col_k = self.dist[:, k].copy()
        via_k = col_k[:, self.cp.newaxis] + row_k_gpu[self.cp.newaxis, :]
        self.cp.minimum(self.dist, via_k, out=self.dist)
        self.cp.cuda.Stream.null.synchronize()

    def obtener_resultado(self) -> Tuple[int, np.ndarray]:
        return self.inicio, self.cp.asnumpy(self.dist)


def _crear_actores_gpu(dist: np.ndarray, num_actores: int) -> list:
    n = dist.shape[0]
    tamano_bloque = max(1, n // num_actores)
    actores = []
    for inicio in range(0, n, tamano_bloque):
        fin = min(inicio + tamano_bloque, n)
        bloque = dist[inicio:fin, :].copy()
        actor = GPUFilasActor.remote(bloque, inicio)
        actores.append((actor, inicio, fin))
    return actores


def floyd_warshall_gpu_ray_multi(
    distancias: np.ndarray,
    num_actores: int = 8,
    inicializar: bool = True,
) -> Tuple[np.ndarray, dict]:
    """
    Floyd-Warshall multi-GPU con actores Ray (un actor por GPU).

    Args:
        distancias: Matriz de adyacencia float64 n×n en CPU.
        num_actores: Número de GPU actors (≤ GPUs disponibles en el cluster).
        inicializar: Si True, inicializa Ray si no está activo.

    Returns:
        Tupla (matriz_resultado_cpu, métricas_dict).
    """
    try:
        import cupy as cp  # noqa: F401
    except ImportError:
        raise RuntimeError("CuPy no disponible.")

    if inicializar and not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    gpus_disponibles = int(ray.cluster_resources().get("GPU", 0))
    num_actores_efectivos = min(num_actores, gpus_disponibles, distancias.shape[0])

    if num_actores_efectivos < num_actores:
        logger.warning(
            "num_actores reducido a %d (GPUs disponibles en cluster: %d)",
            num_actores_efectivos, gpus_disponibles,
        )

    n = distancias.shape[0]

    t_setup_inicio = time.perf_counter()
    actores_info = _crear_actores_gpu(distancias, num_actores_efectivos)
    t_setup = time.perf_counter() - t_setup_inicio

    tiempos_por_k: list = []
    t_calculo_inicio = time.perf_counter()

    for k in range(n):
        t_k = time.perf_counter()

        # Obtener row_k del actor dueño
        row_k = None
        for actor, inicio, fin in actores_info:
            if inicio <= k < fin:
                row_k = ray.get(actor.obtener_fila.remote(k))
                break

        row_k_ref = ray.put(row_k)

        futures = [
            actor.actualizar_bloque.remote(k, row_k_ref)
            for actor, _, _ in actores_info
        ]
        ray.get(futures)

        tiempos_por_k.append(time.perf_counter() - t_k)

    t_calculo = time.perf_counter() - t_calculo_inicio

    t_rearm_inicio = time.perf_counter()
    resultado = np.empty_like(distancias)
    for actor, inicio, _ in actores_info:
        ini_res, bloque = ray.get(actor.obtener_resultado.remote())
        resultado[ini_res:ini_res + bloque.shape[0], :] = bloque

    # Matar actores explícitamente para liberar GPUs en el pool de Ray.
    # Sin esto, los handles quedan vivos hasta el GC, y los escenarios
    # siguientes con más actores no encuentran GPUs disponibles.
    for actor, _, _ in actores_info:
        ray.kill(actor)
    del actores_info

    t_rearm = time.perf_counter() - t_rearm_inicio

    t_total = t_setup + t_calculo + t_rearm
    tiempos_arr = np.array(tiempos_por_k)

    metricas = {
        "algoritmo": "gpu_ray_multi",
        "n": n,
        "num_actores": num_actores_efectivos,
        "gpus_utilizadas": num_actores_efectivos,
        "tiempo_total_s": float(t_total),
        "tiempo_calculo_s": float(t_calculo),
        "tiempo_setup_s": float(t_setup),
        "tiempo_rearmado_s": float(t_rearm),
        "overhead_ray_s": float(t_setup + t_rearm),
        "tiempo_promedio_k_s": float(tiempos_arr.mean()),
        "tiempo_std_k_s": float(tiempos_arr.std()),
        "tiempo_min_k_s": float(tiempos_arr.min()),
        "tiempo_max_k_s": float(tiempos_arr.max()),
    }

    logger.info(
        "Floyd-Warshall GPU multi (%d GPUs): n=%d, cómputo=%.4fs",
        num_actores_efectivos, n, t_calculo,
    )
    return resultado, metricas

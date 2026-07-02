"""
Floyd-Warshall bloqueado multi-GPU coordinado por Ray.

Diseño: p actores Ray, cada uno con num_gpus=1.
Cada actor mantiene su partición de filas en VRAM (local_rows × stride).

Por iteración de bloque b (b = 0..num_block-1):

    Phase 1 (solo el owner del bloque-fila b):
        Actualiza el bloque pivote (b, b) en su VRAM local.
        Comunica: pivot B×B al object store.

    Phase 2 row (solo el owner):
        Actualiza todos los bloques-fila b = (b, jb≠b) en su partición
        usando el pivote ya convergido. Comunica: row_b (B×stride) al object store.

    Phase 2 col (todos los actores, solo los propios):
        Cada actor actualiza sus bloques de columna b = (ib≠b, b) localmente,
        usando el pivote recibido. Sin comunicación adicional.

    Phase 3 (todos los actores, independiente):
        Cada actor actualiza sus bloques (ib≠b, jb≠b) usando:
          - col_b: su propia columna b (recién actualizada en Ph2 col)
          - row_b: recibida del owner (broadcast del object store)
        Sin comunicación adicional.

Complejidad de comunicación: O(B·n) por iteración de bloque, O(n²) total.
Idéntica al naïve multi-GPU en total de bytes, pero con mejor aprovechamiento
de shared memory (L1) dentro de cada GPU: B iteraciones de cómputo por cada
carga desde DRAM → relación cómputo/memoria B× mayor que el naïve.

Nota sobre la implementación CuPy sin kernels CUDA propios:
    Los bucles internos de B=16 iteraciones operan sobre slices completos
    del array CuPy (no bucles Python sobre elementos), lo que permite al
    compilador JIT de CuPy emitir código CUDA eficiente por operación.
    El costo de 16 llamadas a CuPy por fase (en lugar de 1 kernel) es
    despreciable frente al tiempo de transferencia PCIe inter-GPU.
"""
import logging
import time
from typing import Tuple

import numpy as np
import ray

logger = logging.getLogger(__name__)

BLOCK_SIZE = 16


@ray.remote(num_gpus=1)
class GPUBloqueadoActor:
    """
    Actor Ray con partición de filas del bloque padded en VRAM.

    Almacena dist_local[local_rows × stride] donde stride = num_block * B.
    Las columnas son globales (el actor tiene todas las columnas pero solo
    sus filas), lo que permite que Phase 2 col y Phase 3 se ejecuten
    localmente sin consultar a otros actores.
    """

    def __init__(
        self,
        bloque_local: np.ndarray,
        inicio_fila: int,
        n_global: int,
        stride: int,
        block_size: int,
    ):
        import os
        # Caché JIT por proceso: evita deadlock cuando múltiples actores
        # compilan kernels CUDA simultáneamente en el mismo directorio.
        os.environ['CUPY_CACHE_DIR'] = f'/tmp/cupy_cache_{os.getpid()}'
        import cupy as cp
        self.cp = cp
        self.B = block_size
        self.n = n_global
        self.stride = stride
        self.local_rows = bloque_local.shape[0]
        self.inicio_fila = inicio_fila
        self.inicio_bid = inicio_fila // block_size
        self.num_local_blocks = bloque_local.shape[0] // block_size
        self.dist = cp.asarray(bloque_local, dtype=cp.float64)

    def run_ph1_ph2row(self, b: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solo el owner de bloque-fila b llama esto.
        Ejecuta Phase 1 (pivote) y Phase 2 row (fila b de bloques).
        Retorna (pivot B×B, row_b B×stride) como numpy para broadcast.
        """
        cp = self.cp
        B = self.B
        lb = b - self.inicio_bid  # índice de bloque local

        actual_B = min(B, self.n - b * B)  # columnas reales en el bloque pivote

        # ── Phase 1: actualizar bloque pivote (lb*B, b*B) ─────────────────────
        pivot = self.dist[lb * B:(lb + 1) * B, b * B:(b + 1) * B].copy()
        for u in range(actual_B):
            via = pivot[:, u:u + 1] + pivot[u:u + 1, :]
            cp.minimum(pivot, via, out=pivot)
        self.dist[lb * B:(lb + 1) * B, b * B:(b + 1) * B] = pivot

        # ── Phase 2 row: actualizar toda la fila b de bloques ─────────────────
        # Procesa todas las columnas a la vez: dist[lb*B:(lb+1)*B, :]
        # El pivote es pivot (B×B), ya convergido.
        # via[ty, jx] = pivot[ty, u] + row_copy[u, jx] (propagación secuencial)
        row_copy = self.dist[lb * B:(lb + 1) * B, :].copy()  # B × stride
        for u in range(actual_B):
            via = pivot[:, u:u + 1] + row_copy[u:u + 1, :]
            cp.minimum(row_copy, via, out=row_copy)
        self.dist[lb * B:(lb + 1) * B, :] = row_copy

        cp.cuda.Stream.null.synchronize()
        return cp.asnumpy(pivot), cp.asnumpy(row_copy)

    def run_ph2col_ph3(
        self,
        b: int,
        pivot_np: np.ndarray,
        row_b_np: np.ndarray,
    ) -> None:
        """
        Todos los actores llaman esto con el pivote y row_b del owner.
        Phase 2 col: actualiza localmente los bloques (ib≠b, b) de esta partición.
        Phase 3: actualiza localmente los bloques (ib≠b, jb≠b).
        """
        cp = self.cp
        B = self.B
        n, stride = self.n, self.stride

        actual_B = min(B, n - b * B)
        pivot_gpu = cp.asarray(pivot_np)     # B × B
        row_b_gpu = cp.asarray(row_b_np)     # B × stride

        # ── Phase 2 col: actualizar columna b de bloques (toda la partición) ──
        # dist[:, b*B:(b+1)*B] usando el pivote ya convergido.
        # Excluye el bloque pivote propio (si lo tenemos): el owner ya lo tiene
        # correcto, y para los demás actores, aplicar Ph2 col al bloque pivote
        # sería un acceso fuera de sus filas de todas formas.
        col_b = self.dist[:, b * B:b * B + B].copy()  # local_rows × B
        for u in range(actual_B):
            via = col_b[:, u:u + 1] + pivot_gpu[u:u + 1, :]
            cp.minimum(col_b, via, out=col_b)
        self.dist[:, b * B:b * B + actual_B] = col_b[:, :actual_B]

        # ── Phase 3: actualizar todos los demás bloques ────────────────────────
        # Carga fija: col_b recién calculada (snapshot, no se modifica durante el loop)
        # row_b fija: recibida del owner
        # via[iy, jx] = col_b[iy, u] + row_b[u, jx]  (min sobre u)
        col_b_fixed = self.dist[:, b * B:b * B + B].copy()  # snapshot post-Ph2col
        for u in range(actual_B):
            via = col_b_fixed[:, u:u + 1] + row_b_gpu[u:u + 1, :]
            cp.minimum(self.dist, via, out=self.dist)

        cp.cuda.Stream.null.synchronize()

    def obtener_resultado(self) -> Tuple[int, np.ndarray]:
        return self.inicio_fila, self.cp.asnumpy(self.dist)


def _crear_actores_bloqueados(
    dist_padded: np.ndarray,
    num_actores: int,
    n: int,
    stride: int,
    B: int,
) -> list:
    """Particiona la matriz padded en filas (múltiplos de B) y crea los actores."""
    num_block = stride // B
    actores = []
    for i in range(num_actores):
        # Partición balanceada en unidades de bloques B, garantiza exactamente
        # num_actores actores incluso cuando num_block % num_actores != 0.
        bid_start = (i * num_block) // num_actores
        bid_end = ((i + 1) * num_block) // num_actores
        if bid_start >= bid_end:
            continue
        inicio = bid_start * B
        fin = bid_end * B
        bloque = dist_padded[inicio:fin, :].copy()
        actor = GPUBloqueadoActor.remote(bloque, inicio, n, stride, B)
        actores.append((actor, inicio, fin))
    return actores


def floyd_warshall_gpu_blocked_multi(
    distancias: np.ndarray,
    num_actores: int = 8,
    inicializar: bool = True,
) -> Tuple[np.ndarray, dict]:
    """
    Floyd-Warshall bloqueado multi-GPU con actores Ray (un actor por A100).

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
    num_actores_efectivos = min(num_actores, gpus_disponibles)

    if num_actores_efectivos < num_actores:
        logger.warning(
            "num_actores reducido a %d (GPUs disponibles: %d)",
            num_actores_efectivos, gpus_disponibles,
        )

    B = BLOCK_SIZE
    n = distancias.shape[0]
    num_block = (n + B - 1) // B
    stride = num_block * B

    # Preparar matriz padded en CPU
    dist_padded = np.full((stride, stride), np.inf, dtype=np.float64)
    dist_padded[:n, :n] = distancias

    t_setup_inicio = time.perf_counter()
    actores_info = _crear_actores_bloqueados(
        dist_padded, num_actores_efectivos, n, stride, B
    )
    t_setup = time.perf_counter() - t_setup_inicio

    tiempos_por_bloque: list = []
    t_calculo_inicio = time.perf_counter()

    for b in range(num_block):
        t_b = time.perf_counter()

        b_fila_global = b * B

        # Identificar el actor owner del bloque-fila b
        owner = None
        for actor, inicio, fin in actores_info:
            if inicio <= b_fila_global < fin:
                owner = actor
                break

        # Phase 1 + Phase 2 row (owner)
        pivot_np, row_b_np = ray.get(owner.run_ph1_ph2row.remote(b))

        # Publicar en object store para broadcast eficiente
        pivot_ref = ray.put(pivot_np)
        row_b_ref = ray.put(row_b_np)

        # Phase 2 col + Phase 3 (todos los actores en paralelo)
        futures = [
            actor.run_ph2col_ph3.remote(b, pivot_ref, row_b_ref)
            for actor, _, _ in actores_info
        ]
        ray.get(futures)

        tiempos_por_bloque.append(time.perf_counter() - t_b)

    t_calculo = time.perf_counter() - t_calculo_inicio

    # Rearmar resultado
    t_rearm_inicio = time.perf_counter()
    dist_result_padded = np.empty((stride, stride), dtype=np.float64)
    for actor, inicio, _ in actores_info:
        ini_res, bloque = ray.get(actor.obtener_resultado.remote())
        dist_result_padded[ini_res:ini_res + bloque.shape[0], :] = bloque
    resultado = dist_result_padded[:n, :n]

    # Matar actores explícitamente para liberar GPUs en el pool de Ray.
    for actor, _, _ in actores_info:
        ray.kill(actor)
    del actores_info

    t_rearm = time.perf_counter() - t_rearm_inicio

    t_total = t_setup + t_calculo + t_rearm
    tiempos_arr = np.array(tiempos_por_bloque)

    metricas = {
        "algoritmo": "gpu_blocked_multi",
        "n": n,
        "num_actores": num_actores_efectivos,
        "gpus_utilizadas": num_actores_efectivos,
        "block_size": B,
        "num_bloques": num_block,
        "tiempo_total_s": float(t_total),
        "tiempo_calculo_s": float(t_calculo),
        "tiempo_setup_s": float(t_setup),
        "tiempo_rearmado_s": float(t_rearm),
        "overhead_ray_s": float(t_setup + t_rearm),
        "tiempo_promedio_bloque_s": float(tiempos_arr.mean()),
        "tiempo_std_bloque_s": float(tiempos_arr.std()),
        "tiempo_min_bloque_s": float(tiempos_arr.min()),
        "tiempo_max_bloque_s": float(tiempos_arr.max()),
    }

    logger.info(
        "Floyd-Warshall GPU bloqueado multi (%d GPUs): n=%d, B=%d, cómputo=%.4fs",
        num_actores_efectivos, n, B, t_calculo,
    )
    return resultado, metricas

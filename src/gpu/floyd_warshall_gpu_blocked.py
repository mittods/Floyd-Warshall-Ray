"""
Floyd-Warshall bloqueado sobre GPU mediante CuPy RawKernel.

Algoritmo de tres fases (Huang & Cheng, blocked APSP):

    Phase 1 — pivote (1 bloque, auto-dependiente):
        Actualiza el bloque diagonal (b, b) usando shared memory.
        Equivale a correr FW completo dentro del tile.

    Phase 2 — semi-dependiente (fila b y columna b):
        Actualiza los bloques (b, j≠b) y (i≠b, b) usando el pivote
        ya convergido de Phase 1. Cada bloque se actualiza con sí mismo
        y con el bloque pivote en shared memory.

    Phase 3 — independiente (todo lo demás):
        Actualiza (i≠b, j≠b) en paralelo total, leyendo la fila b y la
        columna b (salidas de Phase 2) desde shared memory. Sin dependencias
        cruzadas entre bloques de esta fase.

Ventaja frente al kernel naïve:
    El naïve lee dist[:, k] y dist[k, :] desde DRAM global n veces (una
    por iteración k). El bloqueado carga el tile completo en shared memory
    (L1, ~100× más rápida) y amortiza el costo de DRAM sobre BLOCK_SIZE
    iteraciones por carga.

BLOCK_SIZE = 16 → tile float64 ocupa 16×16×8 = 2 KB en shared memory.
La A4000 tiene 48 KB por SM, con margen para múltiples bloques activos.
"""
import logging
import time
from typing import Tuple

import numpy as np

logger = logging.getLogger(__name__)

BLOCK_SIZE = 16

# ── Código fuente de los tres kernels CUDA ───────────────────────────────────
_KERNEL_SRC = f"""
#define B {BLOCK_SIZE}
#define INF __longlong_as_double(0x7ff0000000000000LL)

// ── Phase 1: bloque pivote (auto-dependiente) ────────────────────────────────
// Grid (1,1) · Block (B,B)
extern "C" __global__
void fw_ph1(const int bid, const int n, const int stride, double* __restrict__ dist) {{
    __shared__ double c[B][B];

    const int ty = threadIdx.y, tx = threadIdx.x;
    const int r  = B * bid + ty;
    const int col = B * bid + tx;

    c[ty][tx] = (r < n && col < n) ? dist[r * stride + col] : INF;
    __syncthreads();

    #pragma unroll
    for (int u = 0; u < B; ++u) {{
        __syncthreads();
        if (r < n && col < n) {{
            double via = c[ty][u] + c[u][tx];
            if (via < c[ty][tx]) c[ty][tx] = via;
        }}
        __syncthreads();
    }}

    if (r < n && col < n)
        dist[r * stride + col] = c[ty][tx];
}}

// ── Phase 2: fila y columna del pivote (semi-dependiente) ────────────────────
// Grid (numBlock, 2) · Block (B,B)
//   blockIdx.y == 0  →  fila b:   bloque (b, blockIdx.x)
//   blockIdx.y == 1  →  columna b: bloque (blockIdx.x, b)
extern "C" __global__
void fw_ph2(const int bid, const int n, const int stride, double* __restrict__ dist) {{
    if (blockIdx.x == bid) return;

    const int ty = threadIdx.y, tx = threadIdx.x;
    __shared__ double pivot[B][B];
    __shared__ double curr[B][B];

    // Cargar bloque pivote
    const int pr = B * bid + ty, pc = B * bid + tx;
    pivot[ty][tx] = (pr < n && pc < n) ? dist[pr * stride + pc] : INF;

    // Coordenadas del bloque actual
    int r, col;
    if (blockIdx.y == 0) {{ r = B * bid + ty;       col = B * blockIdx.x + tx; }}
    else                  {{ r = B * blockIdx.x + ty; col = B * bid + tx; }}

    double val = (r < n && col < n) ? dist[r * stride + col] : INF;
    curr[ty][tx] = val;
    __syncthreads();

    if (r < n && col < n) {{
        #pragma unroll
        for (int u = 0; u < B; ++u) {{
            double via = (blockIdx.y == 0)
                ? pivot[ty][u] + curr[u][tx]   // fila:   dist[i][u_pivot] + dist[u_pivot][j]
                : curr[ty][u]  + pivot[u][tx];  // col:    dist[i][u_pivot] + dist[u_pivot][j]
            if (via < val) val = via;
            __syncthreads();
            curr[ty][tx] = val;
            __syncthreads();
        }}
        dist[r * stride + col] = val;
    }}
}}

// ── Phase 3: bloques independientes ─────────────────────────────────────────
// Grid (numBlock, numBlock) · Block (B,B)
// Omite bloques en fila bid y columna bid (actualizados en Phase 2)
extern "C" __global__
void fw_ph3(const int bid, const int n, const int stride, double* __restrict__ dist) {{
    if (blockIdx.x == bid || blockIdx.y == bid) return;

    const int ty = threadIdx.y, tx = threadIdx.x;

    // Bloque de fila (bid, blockIdx.x) — salida de Phase 2
    __shared__ double row_blk[B][B];
    const int rr = B * bid + ty, rc = B * blockIdx.x + tx;
    row_blk[ty][tx] = (rr < n && rc < n) ? dist[rr * stride + rc] : INF;

    // Bloque de columna (blockIdx.y, bid) — salida de Phase 2
    __shared__ double col_blk[B][B];
    const int cr = B * blockIdx.y + ty, cc = B * bid + tx;
    col_blk[ty][tx] = (cr < n && cc < n) ? dist[cr * stride + cc] : INF;

    __syncthreads();

    const int r   = B * blockIdx.y + ty;
    const int col = B * blockIdx.x + tx;
    if (r < n && col < n) {{
        double val = dist[r * stride + col];
        #pragma unroll
        for (int u = 0; u < B; ++u) {{
            double via = col_blk[ty][u] + row_blk[u][tx];
            if (via < val) val = via;
        }}
        dist[r * stride + col] = val;
    }}
}}
"""


def floyd_warshall_gpu_blocked(
    distancias: np.ndarray,
) -> Tuple[np.ndarray, dict]:
    """
    Ejecuta Floyd-Warshall bloqueado sobre GPU con kernels CuPy RawKernel.

    Args:
        distancias: Matriz de adyacencia float64 n×n en CPU.

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
    num_block = (n + BLOCK_SIZE - 1) // BLOCK_SIZE
    stride = num_block * BLOCK_SIZE  # stride en elementos (filas padded)

    # Compilar kernels (cacheado por CuPy tras la primera llamada)
    module = cp.RawModule(code=_KERNEL_SRC)
    ph1 = module.get_function("fw_ph1")
    ph2 = module.get_function("fw_ph2")
    ph3 = module.get_function("fw_ph3")

    dim_block = (BLOCK_SIZE, BLOCK_SIZE)
    grid_ph1 = (1, 1)
    grid_ph2 = (num_block, 2)
    grid_ph3 = (num_block, num_block)

    # ── Transferencia CPU → GPU con padding a múltiplo de BLOCK_SIZE ─────────
    t_trans_inicio = time.perf_counter()
    dist_gpu = cp.full((stride, stride), np.inf, dtype=cp.float64)
    dist_gpu[:n, :n] = cp.asarray(distancias, dtype=cp.float64)
    cp.cuda.Stream.null.synchronize()
    t_transferencia_ida = time.perf_counter() - t_trans_inicio

    # ── Cómputo bloqueado ─────────────────────────────────────────────────────
    tiempos_por_bloque: list = []
    t_calculo_inicio = time.perf_counter()

    for b in range(num_block):
        t_b = time.perf_counter()

        args = (np.int32(b), np.int32(n), np.int32(stride), dist_gpu)
        ph1(grid_ph1, dim_block, args)
        cp.cuda.Stream.null.synchronize()

        ph2(grid_ph2, dim_block, args)
        cp.cuda.Stream.null.synchronize()

        ph3(grid_ph3, dim_block, args)
        cp.cuda.Stream.null.synchronize()

        tiempos_por_bloque.append(time.perf_counter() - t_b)

    t_calculo = time.perf_counter() - t_calculo_inicio

    # ── Transferencia GPU → CPU (solo la región válida) ───────────────────────
    t_vuelta_inicio = time.perf_counter()
    resultado = cp.asnumpy(dist_gpu[:n, :n])
    cp.cuda.Stream.null.synchronize()
    t_transferencia_vuelta = time.perf_counter() - t_vuelta_inicio

    t_transferencia = t_transferencia_ida + t_transferencia_vuelta
    t_total = t_transferencia + t_calculo

    tiempos_arr = np.array(tiempos_por_bloque)
    metricas = {
        "algoritmo": "gpu_blocked",
        "n": n,
        "num_actores": 0,
        "tiempo_total_s": float(t_total),
        "tiempo_calculo_s": float(t_calculo),
        "tiempo_transferencia_s": float(t_transferencia),
        "tiempo_transferencia_ida_s": float(t_transferencia_ida),
        "tiempo_transferencia_vuelta_s": float(t_transferencia_vuelta),
        "overhead_ray_s": 0.0,
        "tiempo_promedio_bloque_s": float(tiempos_arr.mean()),
        "tiempo_std_bloque_s": float(tiempos_arr.std()),
        "tiempo_min_bloque_s": float(tiempos_arr.min()),
        "tiempo_max_bloque_s": float(tiempos_arr.max()),
        "block_size": BLOCK_SIZE,
        "num_bloques": num_block,
    }

    logger.info(
        "Floyd-Warshall GPU bloqueado: n=%d, bloques=%d, cómputo=%.4fs, transferencia=%.4fs",
        n, num_block, t_calculo, t_transferencia,
    )
    return resultado, metricas

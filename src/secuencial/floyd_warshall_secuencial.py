"""
Implementación secuencial optimizada del algoritmo Floyd-Warshall.

Sirve como línea base para comparar contra la versión paralela con Ray.
Utiliza operaciones vectorizadas de NumPy para maximizar el rendimiento
del núcleo de cómputo sin introducir paralelismo explícito.
"""
import time
import logging
import numpy as np
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

INF = np.inf


def inicializar_matriz(n: int, semilla: int = 42, densidad: float = 0.7) -> np.ndarray:
    """
    Genera una matriz de adyacencia aleatoria reproducible de tamaño n×n.

    Args:
        n: Dimensión de la matriz cuadrada.
        semilla: Semilla para reproducibilidad.
        densidad: Fracción de aristas presentes (0.0–1.0).

    Returns:
        Matriz de adyacencia con valores en [1, 100] para aristas existentes
        e INF para las ausentes. Diagonal en 0.
    """
    rng = np.random.default_rng(semilla)
    matriz = rng.integers(1, 101, size=(n, n)).astype(np.float64)
    np.fill_diagonal(matriz, 0.0)

    # Eliminar aristas según la densidad requerida
    mascara_ausente = rng.random((n, n)) > densidad
    mascara_ausente &= ~np.eye(n, dtype=bool)
    matriz[mascara_ausente] = INF

    return matriz


def floyd_warshall_secuencial(
    distancias: np.ndarray,
) -> Tuple[np.ndarray, dict]:
    """
    Ejecuta el algoritmo Floyd-Warshall de forma secuencial.

    Estrategia de implementación:
        Para cada vértice intermedio k (en orden 0..n-1), actualiza
        toda la matriz de forma vectorizada usando broadcasting de NumPy.
        La condición de actualización es:
            dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])
        Expresada vectorialmente:
            dist = min(dist, col_k[:, None] + row_k[None, :])

    La actualización in-place con np.minimum garantiza que se usa la
    versión más reciente de dist[i][k] y dist[k][j] dentro de la
    misma iteración k, lo cual es correcto para este algoritmo.

    Args:
        distancias: Matriz cuadrada de float64 con pesos de aristas.

    Returns:
        Tupla (matriz_resultado, métricas_dict).
        La matriz resultado contiene los caminos mínimos entre todos
        los pares de vértices.
    """
    n = distancias.shape[0]
    dist = distancias.copy()

    tiempos_por_k: list[float] = []
    t_inicio = time.perf_counter()

    for k in range(n):
        t_k = time.perf_counter()

        row_k = dist[k, :]          # Fila k: dist[k][j] para todo j
        col_k = dist[:, k]          # Columna k: dist[i][k] para todo i

        # Ruta alternativa vía k: col_k[i] + row_k[j] para todo (i,j)
        via_k = col_k[:, np.newaxis] + row_k[np.newaxis, :]
        np.minimum(dist, via_k, out=dist)

        tiempos_por_k.append(time.perf_counter() - t_k)

    t_total = time.perf_counter() - t_inicio

    tiempos_arr = np.array(tiempos_por_k)
    metricas = {
        "algoritmo": "secuencial",
        "n": n,
        "tiempo_total_s": float(t_total),
        "tiempo_promedio_k_s": float(tiempos_arr.mean()),
        "tiempo_std_k_s": float(tiempos_arr.std()),
        "tiempo_min_k_s": float(tiempos_arr.min()),
        "tiempo_max_k_s": float(tiempos_arr.max()),
        "tiempos_por_k": tiempos_por_k,
    }

    logger.info(
        "Floyd-Warshall secuencial: n=%d, tiempo=%.4f s",
        n, t_total,
    )
    return dist, metricas

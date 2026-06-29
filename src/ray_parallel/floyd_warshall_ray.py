"""
Implementación paralela del algoritmo Floyd-Warshall utilizando Ray.

Estrategia de paralelización: Actores con particionamiento por filas.

Análisis de dependencias del algoritmo:
    Floyd-Warshall tiene dos niveles de bucle:
      1. Bucle externo sobre k (0..n-1): SECUENCIAL obligatorio.
         La iteración k+1 depende de los resultados de la iteración k,
         ya que dist[i][k] actualizado en la iteración k se usa en k+1.
      2. Bucle interno sobre (i, j): PARALELIZABLE.
         Para un k fijo, todas las actualizaciones dist[i][j] son
         independientes entre sí y pueden calcularse simultáneamente.

Diseño con actores Ray:
    Se crean `num_actores` actores, cada uno manteniendo una partición
    horizontal (bloque de filas) de la matriz de distancias en memoria
    local. Para cada iteración k:
      1. El actor dueño de la fila k la publica en el object store.
      2. Todos los actores reciben la fila k por referencia (sin copia).
      3. Cada actor actualiza su bloque de filas localmente.
      4. No hay transferencia de la matriz completa entre iteraciones.

    Esta estrategia minimiza la comunicación:
      - Por iteración k: O(n) datos (solo la fila k), no O(n²).
      - La columna k la calcula cada actor desde su partición local.

Limitación inherente:
    El bucle sobre k no puede paralelizarse. El speedup teórico máximo
    es proporcional al número de actores que procesan las n² celdas.
    Para matrices pequeñas, el overhead de Ray supera la ganancia.

Recomendación de uso:
    Utilizar para n ≥ 512 con num_actores ≥ 4 para observar speedup
    positivo en el hardware objetivo (Threadripper PRO 5975WX, 32 cores).
"""
import time
import logging
from typing import Optional, Tuple

import numpy as np
import ray

logger = logging.getLogger(__name__)

INF = np.inf


@ray.remote
class FilasActor:
    """
    Actor Ray que almacena y actualiza un bloque horizontal de la
    matriz de distancias.

    Cada instancia es responsable de las filas [inicio_fila, fin_fila)
    de la matriz global de tamaño n×n.
    """

    def __init__(self, bloque: np.ndarray, inicio_fila: int):
        """
        Args:
            bloque: Submatriz de forma (m, n) con las filas a cargo.
            inicio_fila: Índice global de la primera fila del bloque.
        """
        self.dist = bloque.astype(np.float64)
        self.inicio = inicio_fila
        self.fin = inicio_fila + bloque.shape[0]
        self.m = bloque.shape[0]   # Número de filas locales
        self.n = bloque.shape[1]   # Número de columnas (= n global)

    def obtener_fila(self, k: int) -> Optional[np.ndarray]:
        """
        Retorna la fila global k si este actor la posee, None en caso contrario.

        Se invoca para obtener row_k antes de distribuirla a todos los actores.
        """
        idx_local = k - self.inicio
        if 0 <= idx_local < self.m:
            return self.dist[idx_local, :].copy()
        return None

    def actualizar_bloque(self, k: int, row_k: np.ndarray) -> None:
        """
        Actualiza todas las filas locales para la iteración k de Floyd-Warshall.

        Para cada fila i local:
            dist[i][j] = min(dist[i][j], dist[i][k] + row_k[j])

        La columna k se extrae del bloque local (dist[:, k - offset] no aplica;
        k es el índice GLOBAL, y la columna k está en la posición k de cada fila).

        Args:
            k: Índice del vértice intermedio actual (global).
            row_k: Vector fila k de la matriz global (n elementos).
        """
        col_k_local = self.dist[:, k].copy()  # Columna k del bloque local
        via_k = col_k_local[:, np.newaxis] + row_k[np.newaxis, :]
        np.minimum(self.dist, via_k, out=self.dist)

    def obtener_resultado(self) -> Tuple[int, np.ndarray]:
        """Retorna (inicio_fila, bloque) para rearmar la matriz final."""
        return self.inicio, self.dist.copy()


def _crear_actores(
    dist: np.ndarray,
    num_actores: int,
) -> list:
    """Particiona la matriz y crea los actores Ray."""
    n = dist.shape[0]
    tamano_bloque = max(1, n // num_actores)
    actores = []

    for inicio in range(0, n, tamano_bloque):
        fin = min(inicio + tamano_bloque, n)
        bloque = dist[inicio:fin, :].copy()
        actor = FilasActor.remote(bloque, inicio)
        actores.append((actor, inicio, fin))

    return actores


def floyd_warshall_ray(
    distancias: np.ndarray,
    num_actores: int = 4,
    num_cpus_por_actor: float = 1.0,
    inicializar: bool = True,
) -> Tuple[np.ndarray, dict]:
    """
    Ejecuta Floyd-Warshall paralelizado con actores Ray.

    Args:
        distancias: Matriz de adyacencia de float64, tamaño n×n.
        num_actores: Número de actores Ray (= particiones de la matriz).
                     Recomendado: número de núcleos físicos disponibles.
        num_cpus_por_actor: Recursos CPU declarados por actor a Ray.
        inicializar: Si True, inicializa Ray si no está activo.

    Returns:
        Tupla (matriz_resultado, métricas_dict).
    """
    if inicializar and not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    n = distancias.shape[0]

    # Ajustar actores si la matriz es más pequeña que el número solicitado
    num_actores_efectivos = min(num_actores, n)
    if num_actores_efectivos != num_actores:
        logger.warning(
            "num_actores reducido a %d porque n=%d",
            num_actores_efectivos, n,
        )

    # Crear actores con la matriz particionada
    t_setup_inicio = time.perf_counter()
    actores_info = _crear_actores(distancias, num_actores_efectivos)
    actores = [info[0] for info in actores_info]
    t_setup = time.perf_counter() - t_setup_inicio

    tiempos_por_k: list[float] = []
    t_inicio = time.perf_counter()

    for k in range(n):
        t_k = time.perf_counter()

        # Determinar qué actor posee la fila k
        row_k = None
        for actor, inicio, fin in actores_info:
            if inicio <= k < fin:
                row_k = ray.get(actor.obtener_fila.remote(k))
                break

        # Poner row_k en el object store: todos los actores lo leerán
        # desde memoria compartida sin copias adicionales
        row_k_ref = ray.put(row_k)

        # Actualizar todos los bloques en paralelo
        futures = [
            actor.actualizar_bloque.remote(k, row_k_ref)
            for actor, _, _ in actores_info
        ]
        ray.get(futures)

        tiempos_por_k.append(time.perf_counter() - t_k)

    t_calculo = time.perf_counter() - t_inicio

    # Rearmar la matriz final
    t_rearm_inicio = time.perf_counter()
    resultado = np.empty_like(distancias)
    for actor, inicio, fin in actores_info:
        inicio_res, bloque = ray.get(actor.obtener_resultado.remote())
        resultado[inicio_res : inicio_res + bloque.shape[0], :] = bloque
    t_rearm = time.perf_counter() - t_rearm_inicio

    t_total = t_setup + t_calculo + t_rearm

    tiempos_arr = np.array(tiempos_por_k)
    metricas = {
        "algoritmo": "ray_actores",
        "n": n,
        "num_actores": num_actores_efectivos,
        "tiempo_total_s": float(t_total),
        "tiempo_calculo_s": float(t_calculo),
        "tiempo_setup_s": float(t_setup),
        "tiempo_rearmado_s": float(t_rearm),
        "overhead_ray_s": float(t_setup + t_rearm),
        "tiempo_promedio_k_s": float(tiempos_arr.mean()),
        "tiempo_std_k_s": float(tiempos_arr.std()),
        "tiempo_min_k_s": float(tiempos_arr.min()),
        "tiempo_max_k_s": float(tiempos_arr.max()),
        "tiempos_por_k": tiempos_por_k,
    }

    logger.info(
        "Floyd-Warshall Ray: n=%d, actores=%d, tiempo=%.4f s",
        n, num_actores_efectivos, t_total,
    )
    return resultado, metricas

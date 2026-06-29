"""
Script principal de ejecución de benchmarks.

Ejecuta todos los escenarios experimentales definidos, recolectando
métricas del algoritmo y del sistema. Los resultados se almacenan
en JSON y CSV para su posterior análisis.

Uso:
    python -m experimentos.ejecutar_benchmarks [--escenarios GRUPO] [--n N] [--workers W]

Variables de entorno:
    FW_TAMANOS      Tamaños separados por coma, ej: 256,512,1024
    FW_WORKERS      Workers separados por coma, ej: 4,8,16,32
    FW_REPETICIONES Número de repeticiones (default: 10)
    FW_SEMILLA      Semilla base (default: 42)
"""
import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import ray

# Agregar el directorio raíz al path
sys.path.insert(0, str(Path(__file__).parents[1]))

from src.secuencial.floyd_warshall_secuencial import (
    floyd_warshall_secuencial,
    inicializar_matriz,
)
from src.ray_parallel.floyd_warshall_ray import floyd_warshall_ray
from src.utils.metricas import (
    MetricasEjecucion,
    calcular_estadisticas,
    filtrar_atipicos_grubbs,
)
from src.utils.monitor import MonitorSistema
from src.utils.exportador import exportar_resultados
from experimentos.config import CONFIGURACION_DEFAULT
from experimentos.escenarios import generar_escenarios, resumir_escenarios, Escenario

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("benchmark")


def ejecutar_escenario(
    escenario: Escenario,
    config=CONFIGURACION_DEFAULT,
    verificar_correctitud: bool = True,
) -> list:
    """
    Ejecuta todas las repeticiones de un escenario y retorna la lista
    de registros de métricas.

    Args:
        escenario: Especificación del escenario a ejecutar.
        config: Configuración global del experimento.
        verificar_correctitud: Si True, compara con la solución de referencia
                               en la primera repetición.

    Returns:
        Lista de dicts con métricas de cada repetición.
    """
    registros: list = []
    logger.info(
        "Iniciando escenario: %s (%d repeticiones)",
        escenario.id, escenario.num_repeticiones,
    )

    # Referencia para verificación de correctitud (solo una vez por escenario)
    dist_referencia = None

    for rep in range(escenario.num_repeticiones):
        semilla = config.semilla_base + rep
        matriz = inicializar_matriz(
            escenario.n,
            semilla=semilla,
            densidad=config.densidad_grafo,
        )

        logger.info(
            "  Rep %d/%d: n=%d, algoritmo=%s, actores=%d",
            rep + 1, escenario.num_repeticiones,
            escenario.n, escenario.algoritmo, escenario.num_actores,
        )

        with MonitorSistema(intervalo_s=config.intervalo_monitor_s) as monitor:
            if escenario.algoritmo == "secuencial":
                resultado, metricas_alg = floyd_warshall_secuencial(matriz)
            elif escenario.algoritmo == "ray_actores":
                resultado, metricas_alg = floyd_warshall_ray(
                    matriz,
                    num_actores=escenario.num_actores,
                    inicializar=False,  # Ray ya inicializado en main
                )
            else:
                raise ValueError(f"Algoritmo desconocido: {escenario.algoritmo}")

        # Verificar correctitud en la primera repetición del primer escenario
        if verificar_correctitud and rep == 0:
            if escenario.algoritmo == "secuencial":
                dist_referencia = resultado.copy()
            elif dist_referencia is not None:
                if not np.allclose(resultado, dist_referencia, equal_nan=False, rtol=1e-9):
                    logger.error(
                        "FALLO DE CORRECTITUD en escenario %s, rep %d",
                        escenario.id, rep,
                    )
                else:
                    logger.info("  Correctitud verificada.")

        # Construir registro completo
        metricas_sistema = monitor.obtener_metricas()
        registro = {
            "id_escenario": escenario.id,
            "grupo": escenario.grupo,
            "descripcion": escenario.descripcion,
            "repeticion": rep,
            "semilla": semilla,
            **metricas_alg,
            **metricas_sistema,
        }
        registros.append(registro)
        logger.info(
            "  → tiempo_total=%.4f s",
            metricas_alg["tiempo_total_s"],
        )

    return registros


def agregar_repeticiones(registros: list) -> dict:
    """
    Agrega las métricas de múltiples repeticiones de un escenario.

    Aplica el test de Grubbs para eliminar outliers antes de calcular
    las estadísticas descriptivas.
    """
    if not registros:
        return {}

    tiempos = [r["tiempo_total_s"] for r in registros]
    tiempos_filtrados = filtrar_atipicos_grubbs(tiempos)
    stats = calcular_estadisticas(tiempos_filtrados)

    outliers_eliminados = len(tiempos) - len(tiempos_filtrados)
    if outliers_eliminados > 0:
        logger.info(
            "Grubbs eliminó %d outlier(s) de %d muestras.",
            outliers_eliminados, len(tiempos),
        )

    primer = registros[0]
    return {
        "id_escenario": primer["id_escenario"],
        "grupo": primer["grupo"],
        "n": primer["n"],
        "algoritmo": primer["algoritmo"],
        "num_actores": primer.get("num_actores", 0),
        "n_repeticiones_totales": len(tiempos),
        "n_repeticiones_validas": len(tiempos_filtrados),
        "outliers_eliminados": outliers_eliminados,
        **{f"tiempo_{k}": v for k, v in stats.items()},
        "ram_pico_mb_promedio": float(np.mean([r.get("ram_pico_mb", 0) for r in registros])),
        "cpu_uso_promedio_pct_promedio": float(np.mean([r.get("cpu_uso_promedio_pct", 0) for r in registros])),
        "gpu_uso_pct_promedio": float(np.mean([r.get("gpu_uso_pct", 0) for r in registros])),
        "energia_total_j_promedio": float(np.mean([r.get("energia_total_j", 0) for r in registros])),
    }


def calcular_speedup(
    resultados_agregados: list,
) -> list:
    """
    Calcula el speedup de cada configuración Ray respecto al secuencial
    del mismo tamaño n.

    Args:
        resultados_agregados: Lista de dicts de resultados agregados.

    Returns:
        Lista de dicts con speedup y eficiencia añadidos.
    """
    # Índice de tiempos secuenciales por n
    tiempos_seq = {
        r["n"]: r["tiempo_media"]
        for r in resultados_agregados
        if r["algoritmo"] == "secuencial"
    }

    for r in resultados_agregados:
        n = r["n"]
        t_ray = r.get("tiempo_media", 0)
        t_seq = tiempos_seq.get(n, 0)

        if t_ray > 0 and t_seq > 0:
            r["speedup"] = t_seq / t_ray
        else:
            r["speedup"] = 1.0

        num_actores = r.get("num_actores", 0)
        if num_actores > 0 and r["speedup"] > 0:
            r["eficiencia_paralela"] = r["speedup"] / num_actores
        else:
            r["eficiencia_paralela"] = 1.0

    return resultados_agregados


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ejecuta benchmarks de Floyd-Warshall (secuencial vs. Ray)"
    )
    parser.add_argument(
        "--escenarios",
        nargs="*",
        help="Grupos de escenarios a ejecutar (ej: E1_comparacion E2_escal_fuerte)",
    )
    parser.add_argument(
        "--n",
        type=int,
        nargs="*",
        help="Filtrar por tamaño(s) de matriz",
    )
    parser.add_argument(
        "--workers",
        type=int,
        nargs="*",
        help="Filtrar por número(s) de workers",
    )
    parser.add_argument(
        "--sin-verificacion",
        action="store_true",
        help="Omitir verificación de correctitud",
    )
    parser.add_argument(
        "--solo-mostrar",
        action="store_true",
        help="Mostrar escenarios a ejecutar sin ejecutarlos",
    )
    args = parser.parse_args()

    config = CONFIGURACION_DEFAULT
    config.validar()
    config.crear_directorios()

    # Generar escenarios
    todos_escenarios = generar_escenarios(config)

    # Filtrar según argumentos
    escenarios = todos_escenarios
    if args.escenarios:
        escenarios = [e for e in escenarios if e.grupo in args.escenarios]
    if args.n:
        escenarios = [e for e in escenarios if e.n in args.n]
    if args.workers:
        escenarios = [
            e for e in escenarios
            if e.num_actores in args.workers or e.algoritmo == "secuencial"
        ]

    resumen = resumir_escenarios(escenarios)
    logger.info(
        "Escenarios a ejecutar: %d (%d ejecuciones totales)",
        resumen["total_escenarios"],
        resumen["total_ejecuciones"],
    )

    if args.solo_mostrar:
        for e in escenarios:
            print(f"  {e.id}: {e.descripcion} [{e.num_repeticiones} reps]")
        return

    # Inicializar Ray una sola vez
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
        logger.info(
            "Ray inicializado: %d CPUs disponibles",
            int(ray.cluster_resources().get("CPU", 0)),
        )

    # Ejecutar todos los escenarios
    todos_registros: list = []
    todos_agregados: list = []
    t_inicio_total = time.time()

    for i, escenario in enumerate(escenarios):
        logger.info(
            "─── Escenario %d/%d: %s ───",
            i + 1, len(escenarios), escenario.id,
        )
        try:
            registros = ejecutar_escenario(
                escenario,
                config=config,
                verificar_correctitud=not args.sin_verificacion,
            )
            todos_registros.extend(registros)

            # Guardar registros individuales
            exportar_resultados(
                registros,
                nombre_archivo=f"raw_{escenario.id}",
                formato="json",
                directorio=config.dir_resultados,
            )

            # Agregar y guardar
            agregado = agregar_repeticiones(registros)
            todos_agregados.append(agregado)

        except Exception as e:
            logger.error("Error en escenario %s: %s", escenario.id, e, exc_info=True)

    # Calcular speedup global
    todos_agregados = calcular_speedup(todos_agregados)

    # Exportar resultados consolidados
    exportar_resultados(
        todos_registros,
        nombre_archivo="resultados_raw_completos",
        formato="json",
        directorio=config.dir_resultados,
    )
    exportar_resultados(
        todos_registros,
        nombre_archivo="resultados_raw_completos",
        formato="csv",
        directorio=config.dir_resultados,
    )
    exportar_resultados(
        todos_agregados,
        nombre_archivo="resultados_agregados",
        formato="json",
        directorio=config.dir_resultados,
    )
    exportar_resultados(
        todos_agregados,
        nombre_archivo="resultados_agregados",
        formato="csv",
        directorio=config.dir_resultados,
    )

    t_total = time.time() - t_inicio_total
    logger.info(
        "Benchmarks completados en %.1f min. Resultados en %s",
        t_total / 60,
        config.dir_resultados,
    )

    ray.shutdown()


if __name__ == "__main__":
    main()

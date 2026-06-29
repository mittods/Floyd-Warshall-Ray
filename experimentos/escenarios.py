"""
Definición de escenarios experimentales.

Cada escenario es una combinación (n, algoritmo, num_actores) que define
una ejecución única del benchmark. Los escenarios se estructuran para
responder las siguientes preguntas de investigación:

    E1: Comparación directa secuencial vs. Ray para cada tamaño de matriz.
        → Responde: ¿Cuánto acelera Ray en cada escenario?

    E2: Escalabilidad fuerte (strong scaling): tamaño fijo, workers variables.
        → Responde: ¿Cómo evoluciona el speedup al aumentar workers?

    E3: Escalabilidad débil (weak scaling): tamaño proporcional a workers.
        → Responde: ¿Mantiene Ray el tiempo constante al escalar carga y recursos?

    E4: Análisis del overhead: matrices pequeñas donde overhead > beneficio.
        → Responde: ¿Cuál es el tamaño mínimo donde Ray es rentable?

    E5: Carga máxima: mayor tamaño soportado en el hardware disponible.
        → Responde: ¿Cuál es el límite de escalabilidad del sistema?

    E_GPU: Comparación CPU vs GPU (secuencial y con Ray) para cada tamaño.
        → Responde: ¿Cuánto acelera la GPU respecto a CPU? ¿Añade Ray
          valor sobre una GPU ya masivamente paralela, o introduce overhead?
"""
import math
from dataclasses import dataclass
from typing import Optional
from .config import ConfigExperimento, CONFIGURACION_DEFAULT


@dataclass(frozen=True)
class Escenario:
    """Especificación de una única ejecución del benchmark."""
    id: str
    grupo: str                    # E1..E5
    descripcion: str
    n: int
    algoritmo: str                # "secuencial" | "ray_actores"
    num_actores: int              # 0 para secuencial
    num_repeticiones: int


def generar_escenarios(
    config: Optional[ConfigExperimento] = None,
) -> list:
    """
    Genera la lista completa de escenarios a ejecutar.

    Args:
        config: Configuración del experimento. Usa CONFIGURACION_DEFAULT si None.

    Returns:
        Lista de objetos Escenario en orden de ejecución recomendado.
    """
    config = config or CONFIGURACION_DEFAULT
    escenarios: list = []

    # ── E1: Comparación secuencial vs. Ray (mejor configuración de workers) ──
    workers_optimo = max(config.workers_ray)  # Usar máximo workers en E1
    for n in config.tamanos_matriz:
        escenarios.append(Escenario(
            id=f"E1_seq_n{n}",
            grupo="E1_comparacion",
            descripcion=f"Secuencial, n={n}",
            n=n,
            algoritmo="secuencial",
            num_actores=0,
            num_repeticiones=config.num_repeticiones,
        ))
        escenarios.append(Escenario(
            id=f"E1_ray_n{n}_w{workers_optimo}",
            grupo="E1_comparacion",
            descripcion=f"Ray {workers_optimo} actores, n={n}",
            n=n,
            algoritmo="ray_actores",
            num_actores=workers_optimo,
            num_repeticiones=config.num_repeticiones,
        ))

    # ── E2: Escalabilidad fuerte ─────────────────────────────────────────────
    n_fijo_grande = config.tamanos_matriz[-2] if len(config.tamanos_matriz) >= 2 else config.tamanos_matriz[-1]
    for w in config.workers_ray:
        escenarios.append(Escenario(
            id=f"E2_ray_n{n_fijo_grande}_w{w}",
            grupo="E2_escal_fuerte",
            descripcion=f"Escalabilidad fuerte: n={n_fijo_grande}, actores={w}",
            n=n_fijo_grande,
            algoritmo="ray_actores",
            num_actores=w,
            num_repeticiones=config.num_repeticiones,
        ))

    # ── E3: Escalabilidad débil ──────────────────────────────────────────────
    # n_base × sqrt(workers) para escalar la carga cuadráticamente con workers
    # (Floyd-Warshall es O(n³), pero nos interesa el tiempo por n² celdas)
    n_base_debil = 256
    for w in config.workers_ray:
        n_debil = int(n_base_debil * math.sqrt(w))
        # Ajustar a múltiplo de w para partición exacta
        n_debil = max(w, (n_debil // w) * w)
        escenarios.append(Escenario(
            id=f"E3_ray_w{w}_n{n_debil}",
            grupo="E3_escal_debil",
            descripcion=f"Escalabilidad débil: actores={w}, n={n_debil}",
            n=n_debil,
            algoritmo="ray_actores",
            num_actores=w,
            num_repeticiones=config.num_repeticiones,
        ))

    # ── E4: Análisis de overhead (matrices pequeñas) ────────────────────────
    tamanos_pequenos = [n for n in config.tamanos_matriz if n <= 512]
    for n in tamanos_pequenos:
        for w in [1, 2, 4, 8]:
            if w > n:
                continue
            escenarios.append(Escenario(
                id=f"E4_ray_n{n}_w{w}",
                grupo="E4_overhead",
                descripcion=f"Overhead Ray: n={n}, actores={w}",
                n=n,
                algoritmo="ray_actores",
                num_actores=w,
                num_repeticiones=config.num_repeticiones,
            ))

    # ── E5: Carga máxima ─────────────────────────────────────────────────────
    n_maximo = config.tamanos_matriz[-1]
    escenarios.append(Escenario(
        id=f"E5_seq_n{n_maximo}",
        grupo="E5_carga_maxima",
        descripcion=f"Carga máxima secuencial: n={n_maximo}",
        n=n_maximo,
        algoritmo="secuencial",
        num_actores=0,
        num_repeticiones=max(3, config.num_repeticiones // 2),
    ))
    for w in config.workers_ray:
        escenarios.append(Escenario(
            id=f"E5_ray_n{n_maximo}_w{w}",
            grupo="E5_carga_maxima",
            descripcion=f"Carga máxima Ray: n={n_maximo}, actores={w}",
            n=n_maximo,
            algoritmo="ray_actores",
            num_actores=w,
            num_repeticiones=max(3, config.num_repeticiones // 2),
        ))

    # ── E_GPU: Comparación CPU vs GPU ────────────────────────────────────────
    # Se ejecuta solo si la GPU está disponible (verificado en tiempo de ejecución).
    # Incluye las 4 variantes para cada tamaño: cpu_seq, cpu_ray, gpu_seq, gpu_ray.
    workers_optimo = max(config.workers_ray)
    for n in config.tamanos_matriz:
        escenarios.append(Escenario(
            id=f"EGPU_gpu_seq_n{n}",
            grupo="E_GPU",
            descripcion=f"GPU secuencial (CuPy): n={n}",
            n=n,
            algoritmo="gpu_secuencial",
            num_actores=0,
            num_repeticiones=config.num_repeticiones,
        ))
        escenarios.append(Escenario(
            id=f"EGPU_gpu_ray_n{n}",
            grupo="E_GPU",
            descripcion=f"GPU+Ray (CuPy+actor): n={n}",
            n=n,
            algoritmo="gpu_ray",
            num_actores=1,
            num_repeticiones=config.num_repeticiones,
        ))

    return escenarios


def resumir_escenarios(escenarios: list) -> dict:
    """Retorna un resumen estadístico del conjunto de escenarios."""
    grupos = {}
    for e in escenarios:
        grupos.setdefault(e.grupo, []).append(e)

    total_ejecuciones = sum(e.num_repeticiones for e in escenarios)

    return {
        "total_escenarios": len(escenarios),
        "total_ejecuciones": total_ejecuciones,
        "grupos": {g: len(es) for g, es in grupos.items()},
        "tamanos_unicos": sorted(set(e.n for e in escenarios)),
        "workers_unicos": sorted(set(e.num_actores for e in escenarios)),
    }

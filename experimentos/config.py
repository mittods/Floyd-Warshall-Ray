"""
Configuración parametrizable de los experimentos.

Todos los parámetros se pueden sobreescribir mediante variables de entorno
o mediante argumentos de línea de comandos en ejecutar_benchmarks.py.

Justificación estadística de los valores por defecto:
    - NUM_REPETICIONES = 10: Permite calcular el IC al 95% con t-Student
      con 9 grados de libertad (t_0.025,9 = 2.26). Con 10 muestras, la
      amplitud del intervalo es aproximadamente 0.71·σ.
    - TAMANOS_MATRIZ: Se eligen potencias de 2 para facilitar la partición
      entre workers. Comienza en 64 (donde el overhead domina) hasta 4096
      (donde la carga de cómputo domina). Esto permite identificar el
      punto de crossover secuencial↔paralelo.
    - WORKERS_RAY: Se prueban desde 1 (sin paralelismo real) hasta 32
      (número de núcleos físicos del Threadripper PRO 5975WX). Potencias
      de 2 para identificar la curva de escalabilidad fuerte.
"""
import os
from dataclasses import dataclass, field
from pathlib import Path


def _env_int(nombre: str, defecto: int) -> int:
    return int(os.environ.get(nombre, defecto))


def _env_float(nombre: str, defecto: float) -> float:
    return float(os.environ.get(nombre, defecto))


def _env_list_int(nombre: str, defecto: list) -> list:
    val = os.environ.get(nombre)
    if val:
        return [int(x.strip()) for x in val.split(",")]
    return defecto


@dataclass
class ConfigExperimento:
    """Parámetros completos del diseño experimental."""

    # Tamaños de matriz a evaluar (número de vértices)
    tamanos_matriz: list = field(
        default_factory=lambda: _env_list_int(
            "FW_TAMANOS",
            [64, 128, 256, 512, 1024, 2048, 4096],
        )
    )

    # Números de workers/actores Ray a evaluar
    workers_ray: list = field(
        default_factory=lambda: _env_list_int(
            "FW_WORKERS",
            [1, 2, 4, 8, 16, 32],
        )
    )

    # Número de repeticiones independientes por configuración
    num_repeticiones: int = field(
        default_factory=lambda: _env_int("FW_REPETICIONES", 10)
    )

    # Densidad de aristas en la matriz (fracción 0–1)
    densidad_grafo: float = field(
        default_factory=lambda: _env_float("FW_DENSIDAD", 0.7)
    )

    # Semilla base para generación reproducible de matrices
    # Cada repetición usa semilla = SEMILLA_BASE + repeticion
    semilla_base: int = field(
        default_factory=lambda: _env_int("FW_SEMILLA", 42)
    )

    # Nivel de confianza para intervalos de confianza
    nivel_confianza: float = field(
        default_factory=lambda: _env_float("FW_CONFIANZA", 0.95)
    )

    # Intervalo de muestreo del monitor de recursos (segundos)
    intervalo_monitor_s: float = field(
        default_factory=lambda: _env_float("FW_INTERVALO_MONITOR", 0.5)
    )

    # CPUs por actor Ray (declarado a Ray para el scheduler)
    cpus_por_actor: float = field(
        default_factory=lambda: _env_float("FW_CPUS_POR_ACTOR", 1.0)
    )

    # Directorio base del proyecto
    dir_proyecto: Path = field(
        default_factory=lambda: Path(__file__).parents[1]
    )

    # Directorios de salida
    @property
    def dir_resultados(self) -> Path:
        return self.dir_proyecto / "resultados"

    @property
    def dir_graficos(self) -> Path:
        return self.dir_proyecto / "graficos"

    @property
    def dir_metricas(self) -> Path:
        return self.dir_proyecto / "metricas"

    def crear_directorios(self) -> None:
        """Crea todos los directorios de salida necesarios."""
        for d in [self.dir_resultados, self.dir_graficos, self.dir_metricas]:
            d.mkdir(parents=True, exist_ok=True)

    def validar(self) -> None:
        """Verifica que la configuración sea coherente."""
        assert all(n > 0 for n in self.tamanos_matriz), "Tamaños deben ser positivos"
        assert all(w > 0 for w in self.workers_ray), "Workers deben ser positivos"
        assert self.num_repeticiones >= 3, "Mínimo 3 repeticiones"
        assert 0.0 < self.densidad_grafo <= 1.0, "Densidad debe estar en (0, 1]"
        assert 0.0 < self.nivel_confianza < 1.0, "Nivel de confianza en (0, 1)"


# Configuración por defecto utilizada por los scripts
CONFIGURACION_DEFAULT = ConfigExperimento()

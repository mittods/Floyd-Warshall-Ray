"""
Estructura de datos para recolección y agregación de métricas de ejecución.

Consolida las métricas de tiempo del algoritmo con las métricas del sistema
(CPU, memoria, GPU, energía) registradas por MonitorSistema.
"""
import time
import logging
from dataclasses import dataclass, field, asdict
from typing import Optional
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MetricasEjecucion:
    """
    Contenedor de métricas para una ejecución del algoritmo.

    Se construye a partir de los resultados del algoritmo (tiempo)
    y del monitor del sistema (recursos).
    """
    # Identificación del experimento
    id_experimento: str = ""
    algoritmo: str = ""          # "secuencial" | "ray_actores"
    n: int = 0
    num_actores: int = 0
    repeticion: int = 0
    timestamp: str = ""

    # Métricas de tiempo (segundos)
    tiempo_total_s: float = 0.0
    tiempo_calculo_s: float = 0.0
    tiempo_setup_s: float = 0.0
    tiempo_rearmado_s: float = 0.0
    overhead_ray_s: float = 0.0
    tiempo_promedio_k_s: float = 0.0
    tiempo_std_k_s: float = 0.0
    tiempo_min_k_s: float = 0.0
    tiempo_max_k_s: float = 0.0

    # Métricas de CPU
    cpu_uso_promedio_pct: float = 0.0
    cpu_uso_maximo_pct: float = 0.0
    cpu_uso_por_nucleo: list = field(default_factory=list)
    cpu_hilos_activos: int = 0
    cpu_frecuencia_mhz: float = 0.0

    # Métricas de memoria RAM
    ram_pico_mb: float = 0.0
    ram_promedio_mb: float = 0.0
    ram_proceso_pico_mb: float = 0.0
    ram_total_gb: float = 0.0

    # Métricas de GPU (NVIDIA TITAN V)
    gpu_uso_pct: float = 0.0
    gpu_memoria_mb: float = 0.0
    gpu_temperatura_c: float = 0.0
    gpu_potencia_w: float = 0.0
    gpu_energia_j: float = 0.0
    gpu_disponible: bool = False

    # Métricas de energía
    cpu_potencia_w: float = 0.0
    cpu_energia_j: float = 0.0
    energia_total_j: float = 0.0
    potencia_promedio_w: float = 0.0

    # Métricas derivadas (calculadas a posteriori)
    speedup: float = 0.0
    eficiencia_paralela: float = 0.0

    def calcular_speedup(self, tiempo_secuencial_s: float) -> None:
        """Calcula speedup y eficiencia respecto al tiempo secuencial."""
        if self.tiempo_total_s > 0:
            self.speedup = tiempo_secuencial_s / self.tiempo_total_s
        if self.num_actores > 0:
            self.eficiencia_paralela = self.speedup / self.num_actores

    def to_dict(self) -> dict:
        """Serializa a diccionario plano (compatible con pandas/CSV)."""
        d = asdict(self)
        # Aplanar la lista de uso por núcleo
        for i, uso in enumerate(d.get("cpu_uso_por_nucleo", [])):
            d[f"cpu_nucleo_{i:02d}_pct"] = uso
        d.pop("cpu_uso_por_nucleo", None)
        return d

    @classmethod
    def desde_metricas_algoritmo(
        cls,
        metricas_alg: dict,
        id_experimento: str = "",
        repeticion: int = 0,
    ) -> "MetricasEjecucion":
        """
        Crea una instancia a partir del dict retornado por floyd_warshall_*.
        """
        m = cls()
        m.id_experimento = id_experimento
        m.repeticion = repeticion
        m.timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")
        m.algoritmo = metricas_alg.get("algoritmo", "")
        m.n = metricas_alg.get("n", 0)
        m.num_actores = metricas_alg.get("num_actores", 0)
        m.tiempo_total_s = metricas_alg.get("tiempo_total_s", 0.0)
        m.tiempo_calculo_s = metricas_alg.get("tiempo_calculo_s", m.tiempo_total_s)
        m.tiempo_setup_s = metricas_alg.get("tiempo_setup_s", 0.0)
        m.tiempo_rearmado_s = metricas_alg.get("tiempo_rearmado_s", 0.0)
        m.overhead_ray_s = metricas_alg.get("overhead_ray_s", 0.0)
        m.tiempo_promedio_k_s = metricas_alg.get("tiempo_promedio_k_s", 0.0)
        m.tiempo_std_k_s = metricas_alg.get("tiempo_std_k_s", 0.0)
        m.tiempo_min_k_s = metricas_alg.get("tiempo_min_k_s", 0.0)
        m.tiempo_max_k_s = metricas_alg.get("tiempo_max_k_s", 0.0)
        return m


def calcular_estadisticas(tiempos: list[float]) -> dict:
    """
    Calcula estadísticas descriptivas con intervalo de confianza del 95%.

    Utiliza el método de percentiles de Bootstrap para el IC cuando
    la muestra es pequeña (< 30).
    """
    from scipy import stats

    arr = np.array(tiempos)
    n = len(arr)
    media = arr.mean()
    std = arr.std(ddof=1)
    se = std / np.sqrt(n)

    if n >= 2:
        # t de Student bilateral al 95%
        t_critico = stats.t.ppf(0.975, df=n - 1)
        ic_radio = t_critico * se
    else:
        ic_radio = 0.0

    return {
        "n_muestras": n,
        "media": float(media),
        "mediana": float(np.median(arr)),
        "std": float(std),
        "error_estandar": float(se),
        "ic95_inferior": float(media - ic_radio),
        "ic95_superior": float(media + ic_radio),
        "ic95_radio": float(ic_radio),
        "minimo": float(arr.min()),
        "maximo": float(arr.max()),
        "coef_variacion": float(std / media) if media > 0 else 0.0,
    }


def filtrar_atipicos_grubbs(tiempos: list[float], alpha: float = 0.05) -> list[float]:
    """
    Elimina valores atípicos mediante el test de Grubbs (iterativo).

    El test de Grubbs detecta un outlier a la vez. Se itera hasta que
    no haya más outliers significativos al nivel alpha.

    Solo se aplica si hay al menos 3 muestras.
    """
    from scipy import stats

    arr = np.array(tiempos, dtype=float)
    n = len(arr)
    if n < 3:
        return tiempos

    while True:
        media = arr.mean()
        std = arr.std(ddof=1)
        if std == 0:
            break

        # Estadístico G de Grubbs
        G = np.abs(arr - media).max() / std
        idx_max = np.abs(arr - media).argmax()

        # Valor crítico de Grubbs
        n_actual = len(arr)
        t_crit = stats.t.ppf(1 - alpha / (2 * n_actual), df=n_actual - 2)
        G_crit = (
            (n_actual - 1) / np.sqrt(n_actual)
        ) * np.sqrt(t_crit**2 / (n_actual - 2 + t_crit**2))

        if G > G_crit and n_actual > 3:
            logger.debug("Outlier eliminado: %.6f (G=%.3f > G_crit=%.3f)", arr[idx_max], G, G_crit)
            arr = np.delete(arr, idx_max)
        else:
            break

    return arr.tolist()

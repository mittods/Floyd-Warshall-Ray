"""
Generación automática de gráficos para el artículo IEEE.

Lee los resultados agregados y genera figuras en PDF (vectorial)
y PNG (ráster) listas para incluir en el artículo LaTeX.

Gráficos generados:
    speedup_vs_tamano.pdf        Speedup vs. n para varios workers
    speedup_vs_workers.pdf       Speedup vs. workers (escal. fuerte)
    tiempo_vs_tamano.pdf         Tiempo total seq vs. Ray por n
    overhead_vs_tamano.pdf       Overhead absoluto y relativo de Ray
    eficiencia_vs_workers.pdf    Eficiencia paralela vs. workers
    escalabilidad_debil.pdf      Tiempo normalizado (escal. débil)
    consumo_cpu_ram.pdf          CPU% y RAM por configuración
    consumo_energia.pdf          Energía y potencia por configuración
"""
import logging
import sys
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # sin GUI, para entornos headless
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1]))
from src.utils.exportador import consolidar_resultados, DIR_RESULTADOS

logger = logging.getLogger(__name__)

DIR_GRAFICOS = Path(__file__).parents[1] / "graficos"

# Estilo IEEE-compatible
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "lines.linewidth": 1.5,
    "lines.markersize": 5,
})

COLORES = plt.cm.tab10.colors
MARCADORES = ["o", "s", "^", "D", "v", "P", "*"]


def _guardar(fig: plt.Figure, nombre: str) -> None:
    DIR_GRAFICOS.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        ruta = DIR_GRAFICOS / f"{nombre}.{ext}"
        fig.savefig(ruta)
        logger.info("Gráfico guardado: %s", ruta)
    plt.close(fig)


def grafico_speedup_vs_tamano(df: pd.DataFrame) -> None:
    """Speedup vs. tamaño de matriz n para cada cantidad de workers."""
    df_ray = df[df["algoritmo"] == "ray_actores"].copy()
    df_seq = df[df["algoritmo"] == "secuencial"].copy()

    if df_ray.empty or df_seq.empty:
        logger.warning("Sin datos suficientes para speedup_vs_tamano.")
        return

    fig, ax = plt.subplots(figsize=(3.5, 2.6))

    workers_lista = sorted(df_ray["num_actores"].unique())
    for idx, w in enumerate(workers_lista):
        datos_w = df_ray[df_ray["num_actores"] == w].sort_values("n")
        speedup_col = datos_w["speedup"] if "speedup" in datos_w.columns else pd.Series(
            [1.0] * len(datos_w), index=datos_w.index
        )
        ax.plot(
            datos_w["n"],
            speedup_col,
            marker=MARCADORES[idx % len(MARCADORES)],
            color=COLORES[idx % len(COLORES)],
            label=f"{int(w)} actores",
        )

    ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.8, label="Sin aceleración")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Tamaño de matriz $n$")
    ax.set_ylabel("Speedup")
    ax.set_title("Speedup vs. tamaño de matriz")
    ax.legend(loc="upper left", ncol=2)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x)}"))
    _guardar(fig, "speedup_vs_tamano")


def grafico_speedup_vs_workers(df: pd.DataFrame) -> None:
    """Escalabilidad fuerte: speedup vs. workers para un n fijo."""
    df_ray = df[df["algoritmo"] == "ray_actores"].copy()
    if df_ray.empty:
        return

    tamanos = sorted(df_ray["n"].unique())
    fig, ax = plt.subplots(figsize=(3.5, 2.6))

    for idx, n in enumerate(tamanos[-3:]):  # Los 3 mayores tamaños
        datos_n = df_ray[df_ray["n"] == n].sort_values("num_actores")
        speedup = datos_n["speedup"] if "speedup" in datos_n.columns else pd.Series(
            [1.0] * len(datos_n), index=datos_n.index
        )
        ax.plot(
            datos_n["num_actores"],
            speedup,
            marker=MARCADORES[idx],
            color=COLORES[idx],
            label=f"$n={n}$",
        )

    # Línea de speedup ideal (lineal)
    workers_max = df_ray["num_actores"].max()
    w_range = np.array([1, workers_max])
    ax.plot(w_range, w_range, "k--", linewidth=0.8, label="Ideal")

    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xlabel("Número de actores")
    ax.set_ylabel("Speedup")
    ax.set_title("Escalabilidad fuerte")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x)}"))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.1g}"))
    _guardar(fig, "speedup_vs_workers")


def grafico_tiempo_vs_tamano(df: pd.DataFrame) -> None:
    """Tiempo de ejecución total: secuencial vs. Ray."""
    df_seq = df[df["algoritmo"] == "secuencial"].sort_values("n")
    df_ray = df[df["algoritmo"] == "ray_actores"].copy()

    if df_seq.empty:
        return

    fig, ax = plt.subplots(figsize=(3.5, 2.6))

    # Secuencial
    ax.plot(
        df_seq["n"],
        df_seq["tiempo_media"],
        marker="o", color=COLORES[0], label="Secuencial",
    )
    if "tiempo_ic95_radio" in df_seq.columns:
        ax.fill_between(
            df_seq["n"],
            df_seq["tiempo_media"] - df_seq["tiempo_ic95_radio"],
            df_seq["tiempo_media"] + df_seq["tiempo_ic95_radio"],
            alpha=0.2, color=COLORES[0],
        )

    # Ray con mayor número de workers
    workers_max = df_ray["num_actores"].max() if not df_ray.empty else 0
    df_ray_max = df_ray[df_ray["num_actores"] == workers_max].sort_values("n")
    if not df_ray_max.empty:
        ax.plot(
            df_ray_max["n"],
            df_ray_max["tiempo_media"],
            marker="s", color=COLORES[1],
            label=f"Ray ({int(workers_max)} actores)",
        )

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Tamaño de matriz $n$")
    ax.set_ylabel("Tiempo de ejecución (s)")
    ax.set_title("Tiempo de ejecución vs. $n$")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x)}"))
    _guardar(fig, "tiempo_vs_tamano")


def grafico_overhead(df: pd.DataFrame) -> None:
    """Overhead absoluto y relativo de Ray en función de n."""
    df_ray = df[df["algoritmo"] == "ray_actores"].copy()
    if df_ray.empty or "tiempo_overhead_ray_s_promedio" not in df_ray.columns:
        logger.warning("Sin datos de overhead. Columna 'tiempo_overhead_ray_s_promedio' no encontrada.")
        return

    workers_max = df_ray["num_actores"].max()
    datos = df_ray[df_ray["num_actores"] == workers_max].sort_values("n")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 2.6))

    # Overhead absoluto
    ax1.bar(
        range(len(datos)),
        datos["tiempo_overhead_ray_s_promedio"],
        color=COLORES[2], alpha=0.8,
    )
    ax1.set_xticks(range(len(datos)))
    ax1.set_xticklabels(datos["n"].astype(int), rotation=45)
    ax1.set_xlabel("Tamaño $n$")
    ax1.set_ylabel("Overhead (s)")
    ax1.set_title("Overhead absoluto de Ray")

    # Overhead relativo
    pct_overhead = (
        datos["tiempo_overhead_ray_s_promedio"] / datos["tiempo_media"] * 100
    )
    ax2.bar(range(len(datos)), pct_overhead, color=COLORES[3], alpha=0.8)
    ax2.set_xticks(range(len(datos)))
    ax2.set_xticklabels(datos["n"].astype(int), rotation=45)
    ax2.set_xlabel("Tamaño $n$")
    ax2.set_ylabel("Overhead (%)")
    ax2.set_title("Overhead relativo de Ray")

    plt.tight_layout()
    _guardar(fig, "overhead_vs_tamano")


def grafico_eficiencia_paralela(df: pd.DataFrame) -> None:
    """Eficiencia paralela (speedup / workers) vs. número de workers."""
    df_ray = df[df["algoritmo"] == "ray_actores"].copy()
    if df_ray.empty:
        return

    tamanos = sorted(df_ray["n"].unique())
    fig, ax = plt.subplots(figsize=(3.5, 2.6))

    for idx, n in enumerate(tamanos[-3:]):
        datos_n = df_ray[df_ray["n"] == n].sort_values("num_actores")
        if "eficiencia_paralela" in datos_n.columns:
            eficiencia = datos_n["eficiencia_paralela"]
        elif "speedup" in datos_n.columns:
            eficiencia = datos_n["speedup"] / datos_n["num_actores"]
        else:
            eficiencia = pd.Series([1.0 / max(1, w)] * len(datos_n), index=datos_n.index)
        ax.plot(
            datos_n["num_actores"], eficiencia,
            marker=MARCADORES[idx], color=COLORES[idx], label=f"$n={n}$",
        )

    ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.8, label="Eficiencia ideal")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Número de actores")
    ax.set_ylabel("Eficiencia paralela")
    ax.set_title("Eficiencia paralela vs. actores")
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x)}"))
    _guardar(fig, "eficiencia_vs_workers")


def grafico_escalabilidad_debil(df: pd.DataFrame) -> None:
    """Escalabilidad débil: tiempo normalizado vs. workers."""
    df_esc = df[df["grupo"] == "E3_escal_debil"] if "grupo" in df.columns else pd.DataFrame()
    if df_esc.empty:
        logger.warning("Sin datos de escenario E3 (escalabilidad débil).")
        return

    df_esc = df_esc.sort_values("num_actores")
    t_base = df_esc[df_esc["num_actores"] == df_esc["num_actores"].min()]["tiempo_media"].values[0]
    t_normalizado = df_esc["tiempo_media"] / t_base

    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    ax.plot(
        df_esc["num_actores"], t_normalizado,
        marker="o", color=COLORES[0], label="Ray",
    )
    ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.8, label="Ideal")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Número de actores")
    ax.set_ylabel("Tiempo normalizado")
    ax.set_title("Escalabilidad débil")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x)}"))
    _guardar(fig, "escalabilidad_debil")


def grafico_recursos(df: pd.DataFrame) -> None:
    """CPU% promedio y RAM pico para cada configuración."""
    if "cpu_uso_promedio_pct_promedio" not in df.columns:
        logger.warning("Sin métricas de recursos en los resultados.")
        return

    tamanos = sorted(df["n"].unique())
    df_seq = df[df["algoritmo"] == "secuencial"].sort_values("n")
    df_ray = df[df["algoritmo"] == "ray_actores"].copy()
    workers_max = df_ray["num_actores"].max() if not df_ray.empty else 0
    df_ray_max = df_ray[df_ray["num_actores"] == workers_max].sort_values("n")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 2.6))

    x = np.arange(len(tamanos))
    width = 0.35

    # CPU%
    cpu_seq = [
        df_seq[df_seq["n"] == n]["cpu_uso_promedio_pct_promedio"].values[0]
        if not df_seq[df_seq["n"] == n].empty else 0
        for n in tamanos
    ]
    cpu_ray = [
        df_ray_max[df_ray_max["n"] == n]["cpu_uso_promedio_pct_promedio"].values[0]
        if not df_ray_max[df_ray_max["n"] == n].empty else 0
        for n in tamanos
    ]

    ax1.bar(x - width/2, cpu_seq, width, label="Secuencial", color=COLORES[0])
    ax1.bar(x + width/2, cpu_ray, width, label=f"Ray ({int(workers_max)}w)", color=COLORES[1])
    ax1.set_xticks(x)
    ax1.set_xticklabels(tamanos, rotation=45)
    ax1.set_xlabel("Tamaño $n$")
    ax1.set_ylabel("Uso CPU promedio (%)")
    ax1.set_title("Utilización de CPU")
    ax1.legend()

    # RAM
    ram_seq = [
        df_seq[df_seq["n"] == n]["ram_pico_mb_promedio"].values[0]
        if not df_seq[df_seq["n"] == n].empty else 0
        for n in tamanos
    ]
    ram_ray = [
        df_ray_max[df_ray_max["n"] == n]["ram_pico_mb_promedio"].values[0]
        if not df_ray_max[df_ray_max["n"] == n].empty else 0
        for n in tamanos
    ]

    ax2.bar(x - width/2, ram_seq, width, label="Secuencial", color=COLORES[0])
    ax2.bar(x + width/2, ram_ray, width, label=f"Ray ({int(workers_max)}w)", color=COLORES[1])
    ax2.set_xticks(x)
    ax2.set_xticklabels(tamanos, rotation=45)
    ax2.set_xlabel("Tamaño $n$")
    ax2.set_ylabel("RAM pico (MB)")
    ax2.set_title("Consumo de memoria RAM")
    ax2.legend()

    plt.tight_layout()
    _guardar(fig, "consumo_cpu_ram")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    archivo_agregado = DIR_RESULTADOS / "resultados_agregados.json"
    if not archivo_agregado.exists():
        logger.error("No se encontró %s. Ejecute primero make benchmark.", archivo_agregado)
        sys.exit(1)

    df = pd.read_json(archivo_agregado)
    if df.empty or "tiempo_media" not in df.columns:
        logger.error("El archivo de resultados no tiene estadísticas agregadas.")
        sys.exit(1)

    df_agg = df[df["tiempo_media"].notna()].copy()
    for col in ["cpu_uso_promedio_pct_promedio", "ram_pico_mb_promedio",
                "gpu_uso_pct_promedio", "energia_total_j_promedio"]:
        if col in df_agg.columns:
            df_agg[col] = df_agg[col].fillna(0.0)

    DIR_GRAFICOS.mkdir(parents=True, exist_ok=True)

    grafico_speedup_vs_tamano(df_agg)
    grafico_speedup_vs_workers(df_agg)
    grafico_tiempo_vs_tamano(df_agg)
    grafico_overhead(df_agg)
    grafico_eficiencia_paralela(df_agg)
    grafico_escalabilidad_debil(df_agg)
    grafico_recursos(df_agg)

    logger.info("Todos los gráficos generados en %s", DIR_GRAFICOS)


if __name__ == "__main__":
    main()

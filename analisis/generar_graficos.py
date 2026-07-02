"""
Generación automática de gráficos para el artículo IEEE.

Lee los resultados agregados y genera figuras en PDF (vectorial)
y PNG (ráster) listas para incluir en el artículo LaTeX.

Gráficos generados:
    speedup_vs_tamano.pdf        Speedup vs. n (grupo E1, w=32 actores)
    speedup_vs_workers.pdf       Speedup vs. workers (grupos E2 y E5)
    tiempo_vs_tamano.pdf         Tiempo total seq vs. Ray por n (E1)
    eficiencia_vs_workers.pdf    Eficiencia paralela vs. workers (E2 y E5)
    escalabilidad_debil.pdf      Tiempo normalizado (grupo E3)
    consumo_cpu_ram.pdf          CPU% y RAM por configuración (E1)
    consumo_energia.pdf          Energía por configuración (E1, E2, E5)
"""
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1]))
from src.utils.exportador import DIR_RESULTADOS

logger = logging.getLogger(__name__)

DIR_GRAFICOS = Path(__file__).parents[1] / "graficos"

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
    """Speedup vs. n usando solo E1 (una línea por cantidad de workers).

    E1 tiene exactamente un punto por (n, w=32), evitando duplicados
    entre grupos.
    """
    # E1 tiene secuencial y ray_actores para todos los n con w=32
    df_e1 = df[df["grupo"] == "E1_comparacion"].copy()
    df_seq = df_e1[df_e1["algoritmo"] == "secuencial"].sort_values("n")
    df_ray = df_e1[df_e1["algoritmo"] == "ray_actores"].sort_values("n")

    if df_seq.empty or df_ray.empty:
        logger.warning("Sin datos E1 para speedup_vs_tamano.")
        return

    # Calcular speedup limpio desde E1
    t_seq = df_seq.set_index("n")["tiempo_media"]
    df_ray = df_ray.copy()
    df_ray["speedup_e1"] = df_ray.apply(
        lambda r: t_seq.get(r["n"], float("nan")) / r["tiempo_media"], axis=1
    )

    fig, ax = plt.subplots(figsize=(3.5, 2.6))

    w = int(df_ray["num_actores"].iloc[0])
    ax.plot(
        df_ray["n"], df_ray["speedup_e1"],
        marker="o", color=COLORES[0], label=f"Ray ({w} actores)",
    )
    if "tiempo_ic95_radio" in df_ray.columns:
        ic = df_ray["tiempo_ic95_radio"] / df_ray["tiempo_media"] * df_ray["speedup_e1"]
        ax.fill_between(df_ray["n"], df_ray["speedup_e1"] - ic,
                        df_ray["speedup_e1"] + ic, alpha=0.2, color=COLORES[0])

    ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.8, label="Sin aceleración")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Tamaño de matriz $n$")
    ax.set_ylabel("Speedup $S$")
    ax.set_title("Speedup vs. tamaño de matriz (32 actores)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x)}"))
    _guardar(fig, "speedup_vs_tamano")


def grafico_speedup_vs_workers(df: pd.DataFrame) -> None:
    """Escalabilidad fuerte: speedup vs. workers.

    Usa E2 (n=1024, todos los workers) y E5 (n=2048, todos los workers).
    Cada grupo tiene su propio baseline secuencial → sin contaminación.
    """
    grupos_escal = {
        "E2_escal_fuerte": None,
        "E5_carga_maxima": None,
    }

    fig, ax = plt.subplots(figsize=(3.5, 2.6))

    for idx, grupo in enumerate(grupos_escal):
        df_g = df[df["grupo"] == grupo].copy()
        if df_g.empty:
            continue

        df_seq_g = df_g[df_g["algoritmo"] == "secuencial"]
        df_ray_g = df_g[df_g["algoritmo"] == "ray_actores"].sort_values("num_actores")

        if df_seq_g.empty or df_ray_g.empty:
            continue

        t_ref = df_seq_g["tiempo_media"].iloc[0]
        n_val = int(df_ray_g["n"].iloc[0])
        df_ray_g = df_ray_g.copy()
        df_ray_g["speedup_limpio"] = t_ref / df_ray_g["tiempo_media"]

        ax.plot(
            df_ray_g["num_actores"], df_ray_g["speedup_limpio"],
            marker=MARCADORES[idx], color=COLORES[idx], label=f"$n={n_val}$",
        )

    workers_max = df[df["algoritmo"] == "ray_actores"]["num_actores"].max()
    w_range = np.array([1, workers_max])
    ax.plot(w_range, w_range, "k--", linewidth=0.8, label="Ideal")

    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xlabel("Número de actores")
    ax.set_ylabel("Speedup $S(p)$")
    ax.set_title("Escalabilidad fuerte")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x)}"))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.2g}"))
    _guardar(fig, "speedup_vs_workers")


def grafico_tiempo_vs_tamano(df: pd.DataFrame) -> None:
    """Tiempo de ejecución: secuencial vs. Ray (grupo E1)."""
    df_e1 = df[df["grupo"] == "E1_comparacion"].copy()
    df_seq = df_e1[df_e1["algoritmo"] == "secuencial"].sort_values("n")
    df_ray = df_e1[df_e1["algoritmo"] == "ray_actores"].sort_values("n")

    if df_seq.empty:
        return

    fig, ax = plt.subplots(figsize=(3.5, 2.6))

    ax.plot(df_seq["n"], df_seq["tiempo_media"],
            marker="o", color=COLORES[0], label="Secuencial")
    if "tiempo_ic95_radio" in df_seq.columns:
        ax.fill_between(
            df_seq["n"],
            df_seq["tiempo_media"] - df_seq["tiempo_ic95_radio"],
            df_seq["tiempo_media"] + df_seq["tiempo_ic95_radio"],
            alpha=0.2, color=COLORES[0],
        )

    if not df_ray.empty:
        w = int(df_ray["num_actores"].iloc[0])
        ax.plot(df_ray["n"], df_ray["tiempo_media"],
                marker="s", color=COLORES[1], label=f"Ray ({w} actores)")
        if "tiempo_ic95_radio" in df_ray.columns:
            ax.fill_between(
                df_ray["n"],
                df_ray["tiempo_media"] - df_ray["tiempo_ic95_radio"],
                df_ray["tiempo_media"] + df_ray["tiempo_ic95_radio"],
                alpha=0.2, color=COLORES[1],
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


def grafico_eficiencia_paralela(df: pd.DataFrame) -> None:
    """Eficiencia paralela vs. workers (grupos E2 y E5)."""
    fig, ax = plt.subplots(figsize=(3.5, 2.6))

    for idx, grupo in enumerate(["E2_escal_fuerte", "E5_carga_maxima"]):
        df_g = df[df["grupo"] == grupo].copy()
        df_seq_g = df_g[df_g["algoritmo"] == "secuencial"]
        df_ray_g = df_g[df_g["algoritmo"] == "ray_actores"].sort_values("num_actores")

        if df_seq_g.empty or df_ray_g.empty:
            continue

        t_ref = df_seq_g["tiempo_media"].iloc[0]
        n_val = int(df_ray_g["n"].iloc[0])
        df_ray_g = df_ray_g.copy()
        speedup = t_ref / df_ray_g["tiempo_media"]
        eficiencia = speedup / df_ray_g["num_actores"]

        ax.plot(df_ray_g["num_actores"], eficiencia,
                marker=MARCADORES[idx], color=COLORES[idx], label=f"$n={n_val}$")

    ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.8, label="Ideal")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Número de actores")
    ax.set_ylabel("Eficiencia paralela $E(p)$")
    ax.set_title("Eficiencia paralela vs. actores")
    ax.set_ylim(bottom=0)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x)}"))
    _guardar(fig, "eficiencia_vs_workers")


def grafico_escalabilidad_debil(df: pd.DataFrame) -> None:
    """Escalabilidad débil: tiempo normalizado vs. workers (grupo E3)."""
    df_esc = df[df["grupo"] == "E3_escal_debil"].copy() if "grupo" in df.columns else pd.DataFrame()
    if df_esc.empty:
        logger.warning("Sin datos de escenario E3 (escalabilidad débil).")
        return

    df_esc = df_esc[df_esc["algoritmo"] == "ray_actores"].sort_values("num_actores")
    if df_esc.empty:
        return

    t_base = df_esc["tiempo_media"].iloc[0]
    t_norm = df_esc["tiempo_media"] / t_base

    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    ax.plot(df_esc["num_actores"], t_norm, marker="o", color=COLORES[0], label="Ray")
    ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.8, label="Ideal")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Número de actores")
    ax.set_ylabel("Tiempo normalizado")
    ax.set_title("Escalabilidad débil ($n = 256\\sqrt{p}$)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x)}"))
    _guardar(fig, "escalabilidad_debil")


def grafico_recursos(df: pd.DataFrame) -> None:
    """CPU% promedio y RAM pico (grupo E1)."""
    df_e1 = df[df["grupo"] == "E1_comparacion"].copy()
    df_seq = df_e1[df_e1["algoritmo"] == "secuencial"].sort_values("n")
    df_ray = df_e1[df_e1["algoritmo"] == "ray_actores"].sort_values("n")

    if df_seq.empty or "cpu_uso_promedio_pct_promedio" not in df_seq.columns:
        logger.warning("Sin métricas de recursos en E1.")
        return

    tamanos = sorted(df_seq["n"].unique())
    x = np.arange(len(tamanos))
    width = 0.35
    w_label = int(df_ray["num_actores"].iloc[0]) if not df_ray.empty else "?"

    def _val(subdf, n, col):
        row = subdf[subdf["n"] == n]
        return float(row[col].iloc[0]) if not row.empty else 0.0

    cpu_seq = [_val(df_seq, n, "cpu_uso_promedio_pct_promedio") for n in tamanos]
    cpu_ray = [_val(df_ray, n, "cpu_uso_promedio_pct_promedio") for n in tamanos]
    ram_seq = [_val(df_seq, n, "ram_pico_mb_promedio") for n in tamanos]
    ram_ray = [_val(df_ray, n, "ram_pico_mb_promedio") for n in tamanos]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 2.6))

    ax1.bar(x - width/2, cpu_seq, width, label="Secuencial", color=COLORES[0])
    ax1.bar(x + width/2, cpu_ray, width, label=f"Ray ({w_label}w)", color=COLORES[1])
    ax1.set_xticks(x)
    ax1.set_xticklabels(tamanos, rotation=45)
    ax1.set_xlabel("Tamaño $n$")
    ax1.set_ylabel("CPU promedio (%)")
    ax1.set_title("Utilización de CPU")
    ax1.legend()

    ax2.bar(x - width/2, ram_seq, width, label="Secuencial", color=COLORES[0])
    ax2.bar(x + width/2, ram_ray, width, label=f"Ray ({w_label}w)", color=COLORES[1])
    ax2.set_xticks(x)
    ax2.set_xticklabels(tamanos, rotation=45)
    ax2.set_xlabel("Tamaño $n$")
    ax2.set_ylabel("RAM pico (MB)")
    ax2.set_title("Consumo de memoria RAM")
    ax2.legend()

    plt.tight_layout()
    _guardar(fig, "consumo_cpu_ram")


def grafico_energia(df: pd.DataFrame) -> None:
    """Energía total consumida (J) por configuración.

    Paneles:
      Izq: energía vs n para seq y Ray-32w (grupo E1)
      Der: energía vs workers para n=2048 (grupo E5)
    """
    df_e1 = df[df["grupo"] == "E1_comparacion"].copy()
    df_e5 = df[df["grupo"] == "E5_carga_maxima"].copy()

    col = "energia_total_j_promedio"
    if col not in df.columns:
        logger.warning("Columna '%s' no encontrada.", col)
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 2.6))

    # Panel izquierdo: energía vs n (E1)
    df_seq_e1 = df_e1[df_e1["algoritmo"] == "secuencial"].sort_values("n")
    df_ray_e1 = df_e1[df_e1["algoritmo"] == "ray_actores"].sort_values("n")

    if not df_seq_e1.empty:
        ax1.plot(df_seq_e1["n"], df_seq_e1[col],
                 marker="o", color=COLORES[0], label="Secuencial")
    if not df_ray_e1.empty:
        w = int(df_ray_e1["num_actores"].iloc[0])
        ax1.plot(df_ray_e1["n"], df_ray_e1[col],
                 marker="s", color=COLORES[1], label=f"Ray ({w}w)")

    ax1.set_xscale("log", base=2)
    ax1.set_yscale("log")
    ax1.set_xlabel("Tamaño $n$")
    ax1.set_ylabel("Energía (J)")
    ax1.set_title("Energía vs. tamaño de matriz")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x)}"))

    # Panel derecho: energía vs workers para n=2048 (E5)
    df_seq_e5 = df_e5[df_e5["algoritmo"] == "secuencial"]
    df_ray_e5 = df_e5[df_e5["algoritmo"] == "ray_actores"].sort_values("num_actores")

    if not df_ray_e5.empty:
        ax2.plot(df_ray_e5["num_actores"], df_ray_e5[col],
                 marker="o", color=COLORES[1], label="Ray ($n=2048$)")
        if not df_seq_e5.empty:
            e_seq = float(df_seq_e5[col].iloc[0])
            ax2.axhline(y=e_seq, color=COLORES[0], linestyle="--",
                        linewidth=0.9, label=f"Secuencial ({e_seq:.0f} J)")

    ax2.set_xscale("log", base=2)
    ax2.set_xlabel("Número de actores")
    ax2.set_ylabel("Energía (J)")
    ax2.set_title("Energía vs. actores ($n=2048$)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x)}"))

    plt.tight_layout()
    _guardar(fig, "consumo_energia")


def grafico_gpu_single_comparacion(df: pd.DataFrame) -> None:
    """Tiempo vs n: GPU-seq naïve, GPU-bloqueada y GPU+Ray (1 actor)."""
    NS = [1024, 2048, 4096, 8192]
    series = [
        ("gpu_secuencial", "GPU naïve (CuPy)",      0),
        ("gpu_blocked",    "GPU bloqueada ($B=16$)", 1),
        ("gpu_ray",        "GPU+Ray (1 actor)",      2),
    ]

    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    hay_datos = False

    for alg, label, ci in series:
        sub = df[(df["algoritmo"] == alg) & df["n"].isin(NS)].sort_values("n")
        if sub.empty:
            continue
        hay_datos = True
        ax.plot(sub["n"], sub["tiempo_media"],
                marker=MARCADORES[ci], color=COLORES[ci], label=label)
        if "tiempo_ic95_radio" in sub.columns:
            ax.fill_between(sub["n"],
                            sub["tiempo_media"] - sub["tiempo_ic95_radio"],
                            sub["tiempo_media"] + sub["tiempo_ic95_radio"],
                            alpha=0.15, color=COLORES[ci])

    if not hay_datos:
        logger.warning("Sin datos GPU single para gpu_single_comparacion.")
        return

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Tamaño de matriz $n$")
    ax.set_ylabel("Tiempo de ejecución (s)")
    ax.set_title("GPU single: naïve vs. bloqueado vs. +Ray")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x)}"))
    _guardar(fig, "gpu_single_comparacion")


def grafico_gpu_multi_speedup(df: pd.DataFrame) -> None:
    """Speedup multi-GPU naïve: S(p) = T(1 GPU)/T(p GPUs) vs. num_actores."""
    NS = [1024, 2048, 4096, 8192]
    df_multi = df[df["algoritmo"] == "gpu_ray_multi"].copy()

    if df_multi.empty:
        logger.warning("Sin datos E_GPU_multi para gpu_multi_speedup.")
        return

    fig, ax = plt.subplots(figsize=(3.5, 2.6))

    for ci, n in enumerate(NS):
        sub = df_multi[df_multi["n"] == n].sort_values("num_actores")
        ref = sub[sub["num_actores"] == 1]
        if sub.empty or ref.empty:
            continue
        t_base = ref["tiempo_media"].values[0]
        sub = sub.copy()
        sub["speedup"] = t_base / sub["tiempo_media"]
        ax.plot(sub["num_actores"], sub["speedup"],
                marker=MARCADORES[ci], color=COLORES[ci], label=f"$n={n}$")

    ax.plot([1, 3], [1, 3], "k--", linewidth=0.8, label="Ideal")
    ax.set_xlabel("GPUs (actores Ray)")
    ax.set_ylabel("Speedup $S(p)$")
    ax.set_title("Escalabilidad multi-GPU naïve")
    ax.set_xticks([1, 2, 3])
    ax.legend()
    ax.grid(True, alpha=0.3)
    _guardar(fig, "gpu_multi_speedup")


def grafico_gpu_blocked_multi_speedup(df: pd.DataFrame) -> None:
    """Speedup multi-GPU: naïve (—) vs. bloqueado (- -) para n grandes."""
    NS = [2048, 4096, 8192]

    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    hay_datos = False

    for ci, n in enumerate(NS):
        for alg, estilo, etiq in [
            ("gpu_ray_multi",     "-",  "naïve"),
            ("gpu_blocked_multi", "--", "bloq."),
        ]:
            sub = df[(df["algoritmo"] == alg) & (df["n"] == n)].sort_values("num_actores")
            ref = sub[sub["num_actores"] == 1]
            if sub.empty or ref.empty:
                continue
            hay_datos = True
            t_base = ref["tiempo_media"].values[0]
            sub = sub.copy()
            sub["speedup"] = t_base / sub["tiempo_media"]
            ax.plot(sub["num_actores"], sub["speedup"],
                    linestyle=estilo, marker=MARCADORES[ci], color=COLORES[ci],
                    label=f"$n={n}$ {etiq}")

    if not hay_datos:
        logger.warning("Sin datos multi-GPU para gpu_blocked_multi_speedup.")
        return

    ax.plot([1, 3], [1, 3], "k:", linewidth=0.8, label="Ideal")
    ax.set_xlabel("GPUs (actores Ray)")
    ax.set_ylabel("Speedup $S(p)$")
    ax.set_title("Multi-GPU: naïve (—) vs. bloqueado (- -)")
    ax.set_xticks([1, 2, 3])
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    _guardar(fig, "gpu_blocked_multi_speedup")


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
    grafico_eficiencia_paralela(df_agg)
    grafico_escalabilidad_debil(df_agg)
    grafico_recursos(df_agg)
    grafico_energia(df_agg)
    grafico_gpu_single_comparacion(df_agg)
    grafico_gpu_multi_speedup(df_agg)
    grafico_gpu_blocked_multi_speedup(df_agg)

    logger.info("Todos los gráficos generados en %s", DIR_GRAFICOS)


if __name__ == "__main__":
    main()

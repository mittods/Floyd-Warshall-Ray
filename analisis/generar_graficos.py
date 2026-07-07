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
    consumo_cpu_ram.pdf          RAM pico del proceso, variantes single-GPU
                                 (medición aislada, ver medir_ram_aislado.py)
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

    Combina E2 y E5; itera sobre cada n distinto para evitar puntos dobles.
    El baseline secuencial se busca primero en E5/E2 y luego en E1.
    """
    def _seq_baseline(n_val):
        for grupo in ("E5_carga_maxima", "E2_escal_fuerte", "E1_comparacion"):
            sub = df[(df["grupo"] == grupo) & (df["algoritmo"] == "secuencial") & (df["n"] == n_val)]
            if not sub.empty:
                return float(sub["tiempo_media"].iloc[0])
        return None

    df_ray_all = df[
        df["grupo"].isin(["E2_escal_fuerte", "E5_carga_maxima"]) &
        (df["algoritmo"] == "ray_actores")
    ].copy()

    if df_ray_all.empty:
        return

    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    all_speedups = []

    for ci, n_val in enumerate(sorted(df_ray_all["n"].unique())):
        t_ref = _seq_baseline(n_val)
        if t_ref is None:
            continue
        # Para cada n, usar solo un grupo (E5 > E2) para evitar duplicados
        sub = pd.DataFrame()
        for grp in ("E5_carga_maxima", "E2_escal_fuerte"):
            sub = df_ray_all[(df_ray_all["n"] == n_val) & (df_ray_all["grupo"] == grp)].sort_values("num_actores")
            if not sub.empty:
                break
        if sub.empty:
            continue
        speedup = t_ref / sub["tiempo_media"]
        all_speedups.extend(speedup.tolist())
        ax.plot(sub["num_actores"], speedup.values,
                marker=MARCADORES[ci % len(MARCADORES)],
                color=COLORES[ci % len(COLORES)],
                label=f"$n={n_val}$")

    if not all_speedups:
        plt.close(fig)
        return

    workers_max = int(df_ray_all["num_actores"].max())
    # Trazar la recta ideal para todo el rango x; el ylim la recortará visualmente
    ax.plot([1, workers_max], [1, workers_max], "k--", linewidth=0.8, label="Ideal")
    # Limitar eje y al rango de datos para que la recta ideal no comprima la escala
    y_top = max(all_speedups) * 1.6
    y_bot = min(all_speedups) * 0.75
    ax.set_ylim(bottom=max(y_bot, 0.1), top=y_top)

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
    """Eficiencia paralela vs. workers (grupos E2 y E5).

    Itera sobre cada n distinto para evitar puntos dobles.
    """
    def _seq_baseline(n_val):
        for grupo in ("E5_carga_maxima", "E2_escal_fuerte", "E1_comparacion"):
            sub = df[(df["grupo"] == grupo) & (df["algoritmo"] == "secuencial") & (df["n"] == n_val)]
            if not sub.empty:
                return float(sub["tiempo_media"].iloc[0])
        return None

    df_ray_all = df[
        df["grupo"].isin(["E2_escal_fuerte", "E5_carga_maxima"]) &
        (df["algoritmo"] == "ray_actores")
    ].copy()

    if df_ray_all.empty:
        return

    fig, ax = plt.subplots(figsize=(3.5, 2.6))

    for ci, n_val in enumerate(sorted(df_ray_all["n"].unique())):
        t_ref = _seq_baseline(n_val)
        if t_ref is None:
            continue
        # Para cada n, usar solo un grupo (E5 > E2) para evitar duplicados
        sub = pd.DataFrame()
        for grp in ("E5_carga_maxima", "E2_escal_fuerte"):
            sub = df_ray_all[(df_ray_all["n"] == n_val) & (df_ray_all["grupo"] == grp)].sort_values("num_actores")
            if not sub.empty:
                break
        if sub.empty:
            continue
        speedup = t_ref / sub["tiempo_media"]
        eficiencia = speedup / sub["num_actores"]
        ax.plot(sub["num_actores"], eficiencia.values,
                marker=MARCADORES[ci % len(MARCADORES)],
                color=COLORES[ci % len(COLORES)],
                label=f"$n={n_val}$")

    ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.8, label="Ideal ($E=1$)")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Número de actores")
    ax.set_ylabel("Eficiencia paralela $E(p)$")
    ax.set_title("Eficiencia paralela vs. actores")
    ax.set_ylim(bottom=0, top=1.2)
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
    """RAM pico del proceso para variantes single-GPU.

    A diferencia de las demás figuras, esta NO lee la columna
    ram_proceso_pico_mb_promedio de resultados_agregados.json (el
    parámetro df se ignora a propósito). Esa columna, aunque ya aísla
    el proceso del sistema (RSS vía psutil.Process), sigue contaminada
    por un segundo problema: todos los escenarios de una corrida se
    ejecutan dentro de un único proceso Python (un solo ray.init() en
    ejecutar_benchmarks.main()), y como CPython/glibc/CUDA no le
    devuelven memoria liberada al SO entre escenarios, el "pico" de un
    escenario incluye lo que ya habían reservado los anteriores en el
    mismo proceso.

    Esta figura usa en cambio resultados/ram_aislada/ram_aislada_agregado.json,
    generado por experimentos/medir_ram_aislado.py, donde cada
    combinación (algoritmo, n, repetición) corre en su propio proceso
    del sistema operativo — el RSS medido ahí sí es exclusivamente el
    de esa ejecución. Ver el docstring de ese script para el detalle.
    """
    ruta_aislada = DIR_RESULTADOS / "ram_aislada" / "ram_aislada_agregado.json"
    if not ruta_aislada.exists():
        logger.warning(
            "No se encontró %s. Ejecute primero "
            "'python -m experimentos.medir_ram_aislado' (ver scripts/patagon/"
            "submit_ram_aislada.sbatch) para generar la medición de RAM aislada.",
            ruta_aislada,
        )
        return

    df_ram = pd.read_json(ruta_aislada)

    NS = sorted(df_ram["n"].unique().tolist())
    series = [
        ("gpu_secuencial", "GPU básica",        0),
        ("gpu_blocked",    "GPU segmentada",    1),
        ("gpu_ray",        "GPU+Ray (1 actor)", 2),
    ]

    col_ram = "ram_proceso_pico_mb_promedio"

    def _val(alg, n):
        sub = df_ram[(df_ram["algoritmo"] == alg) & (df_ram["n"] == n)]
        return float(sub[col_ram].iloc[0]) / 1024 if not sub.empty else 0.0

    x = np.arange(len(NS))
    width = 0.25
    fig, ax = plt.subplots(figsize=(5.5, 2.8))

    for i, (alg, label, ci) in enumerate(series):
        vals = [_val(alg, n) for n in NS]
        ax.bar(x + (i - 1) * width, vals, width, label=label, color=COLORES[ci])

    ax.set_xticks(x)
    ax.set_xticklabels(NS)
    ax.set_xlabel("Tamaño $n$")
    ax.set_ylabel("RAM pico del proceso (GB)")
    ax.set_title("Memoria RAM del proceso (medición aislada)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    _guardar(fig, "consumo_cpu_ram")


def grafico_energia(df: pd.DataFrame) -> None:
    """Energía total consumida (J).

    Izq: energía vs n para gpu_secuencial, gpu_blocked, gpu_ray (1 actor).
    Der: energía vs actores para gpu_ray_multi y gpu_blocked_multi (n=8192).
    """
    col = "energia_total_j_promedio"
    if col not in df.columns:
        logger.warning("Columna '%s' no encontrada.", col)
        return

    NS = [1024, 2048, 4096, 8192, 16384]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 2.6))

    # Panel izquierdo: energía vs n — algoritmos GPU de 1 GPU
    single_series = [
        ("gpu_secuencial", None,  "GPU básica",        0),
        ("gpu_blocked",    None,  "GPU segmentada",    1),
        ("gpu_ray",        1,     "GPU+Ray (1 actor)", 2),
    ]
    for alg, actores, label, ci in single_series:
        vals = []
        ns_vals = []
        for n in NS:
            mask = (df["algoritmo"] == alg) & (df["n"] == n)
            if actores is not None:
                mask &= df["num_actores"] == actores
            sub = df[mask]
            if not sub.empty:
                vals.append(float(sub[col].iloc[0]))
                ns_vals.append(n)
        if vals:
            ax1.plot(ns_vals, vals, marker="o", color=COLORES[ci], label=label)

    ax1.set_xscale("log", base=2)
    ax1.set_yscale("log")
    ax1.set_xlabel("Tamaño $n$")
    ax1.set_ylabel("Energía (J)")
    ax1.set_title("Energía vs. tamaño de matriz")
    ax1.legend(fontsize=7)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x)}"))

    # Panel derecho: energía vs actores para n=8192
    N_REF = 8192
    multi_series = [
        ("gpu_ray_multi",        "GPU+Ray multi",    1),
        ("gpu_blocked_multi",    "GPU seg. multi",   2),
    ]
    for alg, label, ci in multi_series:
        sub = df[(df["algoritmo"] == alg) & (df["n"] == N_REF)].sort_values("num_actores")
        if not sub.empty:
            ax2.plot(sub["num_actores"], sub[col], marker="o", color=COLORES[ci], label=label)

    # baseline: gpu_secuencial n=8192
    base = df[(df["algoritmo"] == "gpu_secuencial") & (df["n"] == N_REF)]
    if not base.empty:
        e_base = float(base[col].iloc[0])
        ax2.axhline(y=e_base, color=COLORES[0], linestyle="--",
                    linewidth=0.9, label=f"GPU naïve ({e_base/1000:.1f} kJ)")

    ax2.set_xlabel("Número de actores (GPUs)")
    ax2.set_ylabel("Energía (J)")
    ax2.set_title(f"Energía vs. actores ($n={N_REF}$)")
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x)}"))

    plt.tight_layout()
    _guardar(fig, "consumo_energia")


def grafico_gpu_single_comparacion(df: pd.DataFrame) -> None:
    """Tiempo vs n: GPU-seq naïve, GPU-bloqueada y GPU+Ray (1 actor)."""
    NS = [1024, 2048, 4096, 8192, 16384]
    series = [
        ("gpu_secuencial", "GPU básica (CuPy)",       0),
        ("gpu_blocked",    "GPU segmentada ($B=16$)", 1),
        ("gpu_ray",        "GPU+Ray (1 actor)",       2),
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
    ax.set_title("GPU single: básica vs. segmentada vs. +Ray")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x)}"))
    _guardar(fig, "gpu_single_comparacion")


def grafico_gpu_multi_speedup(df: pd.DataFrame) -> None:
    """Speedup multi-GPU naïve: S(p) = T(1 GPU)/T(p GPUs) vs. num_actores."""
    NS = [1024, 2048, 4096, 8192, 16384]
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
    ax.set_title("Escalabilidad multi-GPU básica")
    ax.set_xticks([1, 2, 3])
    ax.legend()
    ax.grid(True, alpha=0.3)
    _guardar(fig, "gpu_multi_speedup")


def grafico_gpu_blocked_multi_speedup(df: pd.DataFrame) -> None:
    """Speedup multi-GPU: naïve (—) vs. bloqueado (- -) para n grandes."""
    NS = [2048, 4096, 8192, 16384]

    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    hay_datos = False

    for ci, n in enumerate(NS):
        for alg, estilo, etiq in [
            ("gpu_ray_multi",     "-",  "básica"),
            ("gpu_blocked_multi", "--", "seg."),
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
    ax.set_title("Multi-GPU: básica (—) vs. segmentada (- -)")
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
                "ram_proceso_pico_mb_promedio",
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

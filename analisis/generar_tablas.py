"""
Generación automática de tablas LaTeX en formato IEEE.

Lee los resultados agregados del benchmark y genera archivos .tex
que se incluyen directamente en el artículo mediante \\input{}.

Tablas generadas:
    tabla_tiempos.tex        Tiempos secuencial vs. Ray por tamaño n
    tabla_speedup.tex        Speedup y eficiencia paralela
    tabla_escalabilidad.tex  Escalabilidad fuerte y débil
    tabla_overhead.tex       Descomposición del overhead de Ray
    tabla_recursos.tex       CPU, RAM y energía por configuración
"""
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1]))
from src.utils.exportador import consolidar_resultados, DIR_RESULTADOS

logger = logging.getLogger(__name__)

DIR_GRAFICOS = Path(__file__).parents[1] / "graficos"
DIR_RESULTADOS_TEX = Path(__file__).parents[1] / "resultados"


def _formato_tiempo(segundos: float) -> str:
    """Formatea un tiempo en s o ms según magnitud."""
    if segundos < 0.01:
        return f"{segundos * 1000:.2f} ms"
    elif segundos < 10:
        return f"{segundos:.3f} s"
    else:
        return f"{segundos:.1f} s"


def generar_tabla_tiempos(df: pd.DataFrame, ruta_salida: Path) -> None:
    """
    Tabla principal: tiempo secuencial vs. Ray para cada n.

    Columnas: n | T_sec (s) | IC95 sec | T_ray (s) | IC95 ray
    """
    filas_tex = []

    tamanos = sorted(df["n"].unique())
    workers_max = df[df["algoritmo"] == "ray_actores"]["num_actores"].max()

    for n in tamanos:
        fila_seq = df[(df["n"] == n) & (df["algoritmo"] == "secuencial")]
        fila_ray = df[
            (df["n"] == n)
            & (df["algoritmo"] == "ray_actores")
            & (df["num_actores"] == workers_max)
        ]

        if fila_seq.empty:
            continue

        t_seq = fila_seq["tiempo_media"].values[0]
        ic_seq = fila_seq["tiempo_ic95_radio"].values[0] if "tiempo_ic95_radio" in fila_seq else 0.0

        if fila_ray.empty:
            fila = (
                f"  {n} & {_formato_tiempo(t_seq)} $\\pm$ {_formato_tiempo(ic_seq)}"
                f" & -- & -- \\\\"
            )
        else:
            t_ray = fila_ray["tiempo_media"].values[0]
            ic_ray = fila_ray["tiempo_ic95_radio"].values[0] if "tiempo_ic95_radio" in fila_ray else 0.0
            fila = (
                f"  {n} & {_formato_tiempo(t_seq)} $\\pm$ {_formato_tiempo(ic_seq)}"
                f" & {_formato_tiempo(t_ray)} $\\pm$ {_formato_tiempo(ic_ray)} \\\\"
            )

        filas_tex.append(fila)
        filas_tex.append("  \\hline")

    contenido = "\n".join(filas_tex)
    ruta_salida.write_text(contenido, encoding="utf-8")
    logger.info("Tabla de tiempos generada: %s", ruta_salida)


def generar_tabla_speedup(df: pd.DataFrame, ruta_salida: Path) -> None:
    """
    Tabla de speedup y eficiencia paralela.

    Columnas: n | Workers | T_seq (s) | T_ray (s) | Speedup | Eficiencia
    """
    filas_tex = []
    tamanos = sorted(df["n"].unique())

    for n in tamanos:
        fila_seq = df[(df["n"] == n) & (df["algoritmo"] == "secuencial")]
        if fila_seq.empty:
            continue
        t_seq = fila_seq["tiempo_media"].values[0]

        filas_ray = df[
            (df["n"] == n) & (df["algoritmo"] == "ray_actores")
        ].sort_values("num_actores")

        for _, fila in filas_ray.iterrows():
            t_ray = fila["tiempo_media"]
            speedup = fila.get("speedup", t_seq / t_ray if t_ray > 0 else 0)
            eficiencia = fila.get("eficiencia_paralela", speedup / fila["num_actores"])
            w = int(fila["num_actores"])

            linea = (
                f"  {n} & {w} & {_formato_tiempo(t_seq)}"
                f" & {_formato_tiempo(t_ray)}"
                f" & {speedup:.2f} & {eficiencia:.2f} \\\\"
            )
            filas_tex.append(linea)
            filas_tex.append("  \\hline")

    contenido = "\n".join(filas_tex)
    ruta_salida.write_text(contenido, encoding="utf-8")
    logger.info("Tabla de speedup generada: %s", ruta_salida)


def generar_tabla_overhead(df: pd.DataFrame, ruta_salida: Path) -> None:
    """
    Tabla de descomposición del overhead de Ray.

    Columnas: n | Workers | T_calculo (s) | T_overhead (s) | % overhead
    """
    filas_tex = []
    df_ray = df[df["algoritmo"] == "ray_actores"].copy()

    for _, fila in df_ray.sort_values(["n", "num_actores"]).iterrows():
        t_total = fila.get("tiempo_media", 0)
        t_overhead = fila.get("tiempo_overhead_ray_s_promedio", 0)
        t_calculo = t_total - t_overhead
        pct_overhead = (t_overhead / t_total * 100) if t_total > 0 else 0

        linea = (
            f"  {int(fila['n'])} & {int(fila['num_actores'])}"
            f" & {_formato_tiempo(t_calculo)}"
            f" & {_formato_tiempo(t_overhead)}"
            f" & {pct_overhead:.1f}\\% \\\\"
        )
        filas_tex.append(linea)
        filas_tex.append("  \\hline")

    contenido = "\n".join(filas_tex)
    ruta_salida.write_text(contenido, encoding="utf-8")
    logger.info("Tabla de overhead generada: %s", ruta_salida)


def generar_tabla_recursos(df: pd.DataFrame, ruta_salida: Path) -> None:
    """
    Tabla de métricas de recursos: CPU, RAM, energía.

    Columnas: n | Alg. | Workers | CPU% prom | RAM pico (MB) | E_total (J)
    """
    filas_tex = []

    df_valido = df.dropna(subset=["n", "algoritmo"]).copy()
    df_valido["num_actores"] = df_valido["num_actores"].fillna(0)

    for _, fila in df_valido.sort_values(["n", "algoritmo", "num_actores"]).iterrows():
        algoritmo = fila["algoritmo"]
        alg_label = {
            "secuencial": "CPU Sec.",
            "ray_actores": "CPU Ray",
            "gpu_secuencial": "GPU Sec.",
            "gpu_ray": "GPU Ray",
        }.get(algoritmo, algoritmo)
        actores = fila.get("num_actores", 0)
        w = "--" if pd.isna(actores) or actores == 0 else str(int(actores))
        cpu_pct = fila.get("cpu_uso_promedio_pct_promedio", 0)
        ram_mb = fila.get("ram_pico_mb_promedio", 0)
        energia_j = fila.get("energia_total_j_promedio", 0)

        linea = (
            f"  {int(fila['n'])} & {alg_label} & {w}"
            f" & {cpu_pct:.1f}"
            f" & {ram_mb:.0f}"
            f" & {energia_j:.1f} \\\\"
        )
        filas_tex.append(linea)
        filas_tex.append("  \\hline")

    contenido = "\n".join(filas_tex)
    ruta_salida.write_text(contenido, encoding="utf-8")
    logger.info("Tabla de recursos generada: %s", ruta_salida)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    directorio = DIR_RESULTADOS
    df = consolidar_resultados(directorio)

    if df.empty:
        logger.error(
            "No hay resultados en %s. Ejecute primero make benchmark.", directorio
        )
        sys.exit(1)

    # Filtrar solo resultados agregados (no raw individuales)
    if "tiempo_media" not in df.columns:
        logger.error(
            "Los resultados no tienen estadísticas agregadas. "
            "Use resultados_agregados.json."
        )
        sys.exit(1)

    DIR_RESULTADOS_TEX.mkdir(exist_ok=True)

    generar_tabla_tiempos(df, DIR_RESULTADOS_TEX / "tabla_tiempos.tex")
    generar_tabla_speedup(df, DIR_RESULTADOS_TEX / "tabla_speedup.tex")
    generar_tabla_overhead(df, DIR_RESULTADOS_TEX / "tabla_overhead.tex")
    generar_tabla_recursos(df, DIR_RESULTADOS_TEX / "tabla_recursos.tex")

    logger.info("Todas las tablas generadas en %s", DIR_RESULTADOS_TEX)


if __name__ == "__main__":
    main()

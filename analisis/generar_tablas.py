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


def generar_tabla_gpu(df: pd.DataFrame, ruta_salida: Path) -> None:
    """
    Tabla GPU: GPU-seq | GPU-bloqueada | GPU+Ray (1 actor) | S_bloq

    Columnas: n | T_gpu_seq (s) | T_gpu_bloq (s) | T_gpu_ray (s) | S_bloq
    """
    NS = [1024, 2048, 4096, 8192]
    filas_tex = []

    def _fmt(v: float) -> str:
        if np.isnan(v):
            return "--"
        return f"{v:.3f}" if v >= 0.01 else f"{v * 1000:.2f} ms"

    for n in NS:
        def _get(alg: str, filtro_actores: Optional[int] = None) -> float:
            mask = (df["algoritmo"] == alg) & (df["n"] == n)
            if filtro_actores is not None:
                mask &= df["num_actores"] == filtro_actores
            sub = df[mask]
            return float(sub["tiempo_media"].values[0]) if not sub.empty else float("nan")

        t_seq  = _get("gpu_secuencial")
        t_bloq = _get("gpu_blocked")
        t_ray  = _get("gpu_ray", filtro_actores=1)
        s_bloq = t_seq / t_bloq if not (np.isnan(t_seq) or np.isnan(t_bloq) or t_bloq == 0) else float("nan")

        s_str = f"{s_bloq:.2f}$\\times$" if not np.isnan(s_bloq) else "--"
        filas_tex.append(
            f"  {n} & {_fmt(t_seq)} & {_fmt(t_bloq)} & {_fmt(t_ray)} & {s_str} \\\\"
        )
        filas_tex.append("  \\hline")

    ruta_salida.write_text("\n".join(filas_tex), encoding="utf-8")
    logger.info("Tabla GPU generada: %s", ruta_salida)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # Leer solo el archivo de resultados agregados (uno por escenario,
    # no los raw individuales que contaminarían el DataFrame)
    archivo_agregado = DIR_RESULTADOS / "resultados_agregados.json"
    if not archivo_agregado.exists():
        logger.error(
            "No se encontró %s. Ejecute primero make benchmark.", archivo_agregado
        )
        sys.exit(1)

    df = pd.read_json(archivo_agregado)

    if df.empty or "tiempo_media" not in df.columns:
        logger.error(
            "El archivo de resultados no tiene estadísticas agregadas. "
            "Verifique que el benchmark completó al menos un escenario."
        )
        sys.exit(1)

    # Filtrar solo filas con datos válidos y normalizar NaN en recursos
    df = df[df["tiempo_media"].notna()].copy()
    for col in ["cpu_uso_promedio_pct_promedio", "ram_pico_mb_promedio",
                "gpu_uso_pct_promedio", "energia_total_j_promedio"]:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)

    DIR_RESULTADOS_TEX.mkdir(exist_ok=True)

    generar_tabla_tiempos(df, DIR_RESULTADOS_TEX / "tabla_tiempos.tex")
    generar_tabla_speedup(df, DIR_RESULTADOS_TEX / "tabla_speedup.tex")
    generar_tabla_overhead(df, DIR_RESULTADOS_TEX / "tabla_overhead.tex")
    generar_tabla_recursos(df, DIR_RESULTADOS_TEX / "tabla_recursos.tex")
    generar_tabla_gpu(df, DIR_RESULTADOS_TEX / "tabla_gpu.tex")

    logger.info("Todas las tablas generadas en %s", DIR_RESULTADOS_TEX)


if __name__ == "__main__":
    main()

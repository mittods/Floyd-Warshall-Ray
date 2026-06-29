"""
Exportación e importación de resultados experimentales.

Formatos soportados:
    - JSON: para resultados individuales legibles por humanos.
    - CSV: para análisis en pandas / hojas de cálculo.
    - Parquet: para datasets grandes con compresión eficiente.

Todos los resultados se almacenan en el directorio `resultados/`
con nombre de archivo basado en el identificador del experimento.
"""
import json
import logging
from pathlib import Path
from typing import Optional, Union

import pandas as pd

logger = logging.getLogger(__name__)

DIR_RESULTADOS = Path(__file__).parents[2] / "resultados"


def exportar_resultados(
    datos: Union[dict, list],
    nombre_archivo: str,
    formato: str = "json",
    directorio: Optional[Path] = None,
) -> Path:
    """
    Exporta resultados experimentales al directorio de resultados.

    Args:
        datos: Diccionario único o lista de diccionarios con resultados.
        nombre_archivo: Nombre base del archivo (sin extensión).
        formato: "json" | "csv" | "parquet".
        directorio: Directorio destino. Por defecto DIR_RESULTADOS.

    Returns:
        Ruta al archivo generado.
    """
    directorio = directorio or DIR_RESULTADOS
    directorio.mkdir(parents=True, exist_ok=True)

    if isinstance(datos, dict):
        datos = [datos]

    ruta = directorio / f"{nombre_archivo}.{formato}"

    if formato == "json":
        with open(ruta, "w", encoding="utf-8") as f:
            json.dump(datos, f, indent=2, ensure_ascii=False, default=str)

    elif formato == "csv":
        df = pd.DataFrame(datos)
        df.to_csv(ruta, index=False, encoding="utf-8")

    elif formato == "parquet":
        df = pd.DataFrame(datos)
        df.to_parquet(ruta, index=False, compression="snappy")

    else:
        raise ValueError(f"Formato no soportado: {formato}")

    logger.info("Resultados exportados: %s", ruta)
    return ruta


def cargar_resultados(
    nombre_archivo: str,
    formato: str = "json",
    directorio: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Carga resultados previos como DataFrame de pandas.

    Args:
        nombre_archivo: Nombre base del archivo (sin extensión).
        formato: "json" | "csv" | "parquet".
        directorio: Directorio fuente. Por defecto DIR_RESULTADOS.

    Returns:
        DataFrame con los resultados.
    """
    directorio = directorio or DIR_RESULTADOS
    ruta = directorio / f"{nombre_archivo}.{formato}"

    if not ruta.exists():
        raise FileNotFoundError(f"Archivo no encontrado: {ruta}")

    if formato == "json":
        return pd.read_json(ruta)
    elif formato == "csv":
        return pd.read_csv(ruta)
    elif formato == "parquet":
        return pd.read_parquet(ruta)
    else:
        raise ValueError(f"Formato no soportado: {formato}")


def consolidar_resultados(directorio: Optional[Path] = None) -> pd.DataFrame:
    """
    Consolida todos los archivos JSON de resultados en un único DataFrame.

    Útil para generar tablas y gráficos a partir de múltiples ejecuciones.
    """
    directorio = directorio or DIR_RESULTADOS
    registros: list = []

    for archivo in sorted(directorio.glob("*.json")):
        try:
            with open(archivo, encoding="utf-8") as f:
                datos = json.load(f)
            if isinstance(datos, list):
                registros.extend(datos)
            elif isinstance(datos, dict):
                registros.append(datos)
        except Exception as e:
            logger.warning("Error al cargar %s: %s", archivo, e)

    if not registros:
        logger.warning("No se encontraron resultados en %s", directorio)
        return pd.DataFrame()

    return pd.DataFrame(registros)

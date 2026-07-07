"""
Medición aislada de RAM de proceso para las variantes GPU de un solo
dispositivo (gpu_secuencial, gpu_ray, gpu_blocked).

Corrige dos problemas de medición detectados en el benchmark principal
(experimentos/ejecutar_benchmarks.py):

  1. La métrica original (ram_pico_mb) usaba psutil.virtual_memory(),
     es decir, RAM de TODO el sistema — en un nodo Patagón compartido
     queda dominada por ruido ajeno al experimento y no escala con n.

  2. Incluso la RAM del proceso (ram_proceso_pico_mb, vía
     psutil.Process().memory_info().rss) estaba contaminada: todos los
     escenarios de una corrida (E_GPU, E_GPU_blocked, E_GPU_multi, ...)
     se ejecutan dentro de un único proceso Python (un solo ray.init()
     en ejecutar_benchmarks.main()), y como CPython/glibc/CUDA no le
     devuelven memoria liberada al sistema operativo entre escenarios,
     el "pico" medido para un escenario incluye lo que ya habían
     reservado (y no liberado) los escenarios anteriores en ese mismo
     proceso. Por eso, por ejemplo, gpu_blocked -que no usa Ray- podía
     reportar más RAM que gpu_ray: no por costo propio, sino por correr
     después en la misma corrida.

Este script corrige ambos problemas: cada combinación
(algoritmo, n, repetición) se ejecuta en su PROPIO proceso del sistema
operativo, lanzado con subprocess.run. Al terminar cada proceso, el
sistema operativo libera toda su memoria — así el RSS pico medido en
el siguiente caso arranca limpio, sin arrastrar nada de los anteriores.

Solo se conserva la métrica de RAM del proceso. CPU%, uso de GPU y
energía no se miden aquí a propósito (para eso sigue valiendo el
benchmark principal); esos casos NO sufren el problema de acumulación
entre escenarios porque son mediciones diferenciales (fin - inicio)
acotadas exactamente a la ventana de ejecución de cada escenario, no
una foto absoluta del sistema.

Uso:
    # Orquestador: lanza un subproceso nuevo por cada combinación
    # (algoritmo x n x repetición) y agrega los resultados al final.
    python -m experimentos.medir_ram_aislado

    # Solo mostrar los casos que se ejecutarían, sin correr nada:
    python -m experimentos.medir_ram_aislado --solo-mostrar

    # Modo worker (uso interno): ejecuta UN único caso en este proceso
    # y escribe su resultado a disco. Lo invoca el orquestador; no se
    # espera que se llame a mano salvo para depurar un caso puntual.
    python -m experimentos.medir_ram_aislado --worker \\
        --algoritmo gpu_blocked --n 16384 --repeticion 0 --semilla 42
"""
import argparse
import json
import logging
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.utils.exportador import DIR_RESULTADOS  # noqa: E402

logger = logging.getLogger("medir_ram_aislado")

# ── Diseño experimental de esta corrida corregida ────────────────────────────
ALGORITMOS = ["gpu_secuencial", "gpu_ray", "gpu_blocked"]
TAMANOS = [1024, 2048, 4096, 8192, 16384]
REPETICIONES = 2
DENSIDAD_GRAFO = 0.7
SEMILLA_BASE = 42
INTERVALO_MONITOR_S = 0.5

DIR_SALIDA = DIR_RESULTADOS / "ram_aislada"


def _ejecutar_caso(algoritmo: str, n: int, semilla: int, densidad: float) -> dict:
    """
    Genera la matriz y corre el algoritmo pedido en ESTE proceso.

    Debe invocarse siempre desde un proceso recién lanzado (ver el modo
    --worker en main()): nunca se reutiliza un mismo proceso para más
    de un caso, precisamente para que el RSS medido no arrastre memoria
    de ejecuciones anteriores.
    """
    from src.secuencial.floyd_warshall_secuencial import inicializar_matriz
    from src.utils.monitor import MonitorSistema

    matriz = inicializar_matriz(n, semilla=semilla, densidad=densidad)

    with MonitorSistema(intervalo_s=INTERVALO_MONITOR_S) as monitor:
        if algoritmo == "gpu_secuencial":
            from src.gpu.floyd_warshall_gpu import floyd_warshall_gpu
            _, metricas_alg = floyd_warshall_gpu(matriz)
        elif algoritmo == "gpu_ray":
            from src.gpu.floyd_warshall_gpu_ray import floyd_warshall_gpu_ray
            _, metricas_alg = floyd_warshall_gpu_ray(matriz, inicializar=True)
        elif algoritmo == "gpu_blocked":
            from src.gpu.floyd_warshall_gpu_blocked import floyd_warshall_gpu_blocked
            _, metricas_alg = floyd_warshall_gpu_blocked(matriz)
        else:
            raise ValueError(f"Algoritmo desconocido: {algoritmo}")

    metricas_sistema = monitor.obtener_metricas()

    # A propósito NO se guarda cpu_uso, gpu_uso ni energía: el pedido es
    # medir solo RAM, aislada del ruido de acumulación entre escenarios.
    return {
        "algoritmo": algoritmo,
        "n": n,
        "semilla": semilla,
        "tiempo_total_s": float(metricas_alg.get("tiempo_total_s", 0.0)),
        "ram_proceso_pico_mb": float(metricas_sistema["ram_proceso_pico_mb"]),
    }


def _agregar_resultados() -> None:
    """Promedia ram_proceso_pico_mb por (algoritmo, n) sobre las repeticiones."""
    grupos: dict = defaultdict(list)
    for ruta in sorted(DIR_SALIDA.glob("raw_ram_*.json")):
        with open(ruta) as f:
            reg = json.load(f)
        grupos[(reg["algoritmo"], reg["n"])].append(reg["ram_proceso_pico_mb"])

    agregado = []
    for (algoritmo, n), valores in sorted(grupos.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        agregado.append({
            "algoritmo": algoritmo,
            "n": n,
            "num_actores": 1 if algoritmo == "gpu_ray" else 0,
            "n_repeticiones": len(valores),
            "ram_proceso_pico_mb_promedio": float(np.mean(valores)),
            "ram_proceso_pico_mb_min": float(np.min(valores)),
            "ram_proceso_pico_mb_max": float(np.max(valores)),
        })

    ruta_salida = DIR_SALIDA / "ram_aislada_agregado.json"
    ruta_salida.write_text(json.dumps(agregado, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Agregado escrito en %s (%d combinaciones)", ruta_salida, len(agregado))


def _modo_worker(args: argparse.Namespace) -> None:
    resultado = _ejecutar_caso(args.algoritmo, args.n, args.semilla, args.densidad)
    resultado["repeticion"] = args.repeticion

    DIR_SALIDA.mkdir(parents=True, exist_ok=True)
    ruta = DIR_SALIDA / f"raw_ram_{args.algoritmo}_n{args.n}_rep{args.repeticion}.json"
    ruta.write_text(json.dumps(resultado, indent=2, ensure_ascii=False), encoding="utf-8")

    logger.info(
        "Caso completado: %s n=%d rep=%d → ram_proceso_pico_mb=%.1f MB (tiempo=%.2fs)",
        args.algoritmo, args.n, args.repeticion,
        resultado["ram_proceso_pico_mb"], resultado["tiempo_total_s"],
    )


def _modo_orquestador(args: argparse.Namespace) -> None:
    casos = [
        (algoritmo, n, rep)
        for algoritmo in ALGORITMOS
        for n in TAMANOS
        for rep in range(REPETICIONES)
    ]

    if args.solo_mostrar:
        for algoritmo, n, rep in casos:
            print(f"  {algoritmo} n={n} rep={rep} (semilla={SEMILLA_BASE + rep})")
        print(f"\nTotal: {len(casos)} casos, cada uno en un proceso nuevo.")
        return

    DIR_SALIDA.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Lanzando %d casos aislados (%d algoritmos x %d tamaños x %d repeticiones), "
        "1 proceso nuevo por caso.",
        len(casos), len(ALGORITMOS), len(TAMANOS), REPETICIONES,
    )

    proyecto_dir = Path(__file__).parents[1]
    completados, fallidos = 0, []

    for algoritmo, n, rep in casos:
        semilla = SEMILLA_BASE + rep
        cmd = [
            sys.executable, "-m", "experimentos.medir_ram_aislado",
            "--worker",
            "--algoritmo", algoritmo,
            "--n", str(n),
            "--repeticion", str(rep),
            "--semilla", str(semilla),
            "--densidad", str(DENSIDAD_GRAFO),
        ]
        logger.info("→ %s n=%d rep=%d (proceso nuevo)", algoritmo, n, rep)
        proc = subprocess.run(cmd, cwd=str(proyecto_dir))

        if proc.returncode != 0:
            logger.error(
                "  FALLÓ: %s n=%d rep=%d (código de salida %d)",
                algoritmo, n, rep, proc.returncode,
            )
            fallidos.append((algoritmo, n, rep))
        else:
            completados += 1

    logger.info("Casos completados: %d/%d", completados, len(casos))
    if fallidos:
        logger.warning("Casos fallidos (revisar logs arriba): %s", fallidos)

    _agregar_resultados()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Medición aislada de RAM (un proceso por caso) para gpu_secuencial, gpu_ray y gpu_blocked."
    )
    parser.add_argument("--worker", action="store_true", help="Modo interno: ejecuta un único caso en este proceso.")
    parser.add_argument("--algoritmo", choices=ALGORITMOS)
    parser.add_argument("--n", type=int)
    parser.add_argument("--repeticion", type=int)
    parser.add_argument("--semilla", type=int)
    parser.add_argument("--densidad", type=float, default=DENSIDAD_GRAFO)
    parser.add_argument(
        "--solo-mostrar", action="store_true",
        help="Mostrar los casos que se ejecutarían, sin correr nada.",
    )
    args = parser.parse_args()

    if args.worker:
        if args.algoritmo is None or args.n is None or args.repeticion is None or args.semilla is None:
            parser.error("--worker requiere --algoritmo, --n, --repeticion y --semilla")
        _modo_worker(args)
    else:
        _modo_orquestador(args)


if __name__ == "__main__":
    main()

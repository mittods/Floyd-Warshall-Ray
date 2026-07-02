# Floyd-Warshall con Ray

Entorno experimental para el artículo IEEE:

> **"Paralelización del Algoritmo Floyd-Warshall mediante Ray:
> Análisis de Rendimiento y Escalabilidad en Hardware HPC"**
>
> Martín Maza — Instituto de Informática, Universidad Austral de Chile

[![Docker Build](https://github.com/mittods/Floyd-Warshall-Ray/actions/workflows/docker.yml/badge.svg)](https://github.com/mittods/Floyd-Warshall-Ray/actions/workflows/docker.yml)

---

## Requisitos

- [Docker](https://docs.docker.com/get-docker/) ≥ 24
- [Docker Compose](https://docs.docker.com/compose/install/) v2 (incluido en Docker Desktop)
- *(solo para experimentos GPU)* [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) + driver NVIDIA ≥ 525

---

## Inicio rápido con Docker

```bash
git clone https://github.com/mittods/Floyd-Warshall-Ray.git
cd Floyd-Warshall-Ray

# Construir la imagen localmente
make docker-build

# Verificar que todo funciona (test de humo, ~1 min)
make docker-test
```

La imagen también está disponible en GHCR:

```bash
docker pull ghcr.io/mittods/floyd-warshall-ray:latest
```

---

## Despliegue con Docker

### Experimentos CPU

```bash
# Benchmark completo con parámetros por defecto (n=256..2048, workers=1..32)
docker compose run --rm benchmark make benchmark

# Parámetros personalizados
FW_TAMANOS="128,256,512" FW_WORKERS="1,2,4" FW_REPETICIONES=3 \
  docker compose run --rm benchmark make benchmark

# Shell interactivo
docker compose run --rm benchmark bash
```

### Experimentos GPU

Requiere `nvidia-container-toolkit` instalado en el host.

```bash
# Benchmark GPU (variantes básica, segmentada y multi-GPU)
docker compose --profile gpu run --rm benchmark-gpu \
  python -m experimentos.ejecutar_benchmarks \
    --escenarios E_GPU E_GPU_blocked E_GPU_multi E_GPU_blocked_multi

# Shell interactivo con GPU
docker compose --profile gpu run --rm benchmark-gpu bash
```

### Análisis (gráficos y tablas)

Los resultados se generan en `./resultados/` y los gráficos en `./graficos/`
mediante volúmenes montados.

```bash
docker compose run --rm analisis
```

O manualmente desde un shell:

```bash
docker compose run --rm benchmark bash -c "
  python -m analisis.generar_graficos &&
  python -m analisis.generar_tablas
"
```

### Test de humo

Verifica que la imagen está correctamente configurada sin necesitar GPU:

```bash
bash scripts/test_docker.sh                              # imagen local
bash scripts/test_docker.sh ghcr.io/mittods/floyd-warshall-ray:latest
```

El test ejecuta el benchmark CPU con `n=128` y 1 repetición (~30 s) y
comprueba que `resultados/resultados_agregados.json` se genera correctamente.

---

## Variables de entorno

| Variable | Por defecto | Descripción |
|---|---|---|
| `FW_TAMANOS` | `256,512,1024,2048` | Tamaños de matriz n separados por coma |
| `FW_WORKERS` | `1,2,4,8,16,32` | Número de actores Ray |
| `FW_REPETICIONES` | `5` | Repeticiones por configuración |
| `FW_SEMILLA` | `42` | Semilla para matrices aleatorias |
| `FW_DENSIDAD` | `0.7` | Densidad del grafo (fracción de aristas) |

---

## Estructura del repositorio

```
Floyd-Warshall-Ray/
├── src/
│   ├── secuencial/          # Implementación CPU secuencial (línea base)
│   ├── ray_parallel/        # Floyd-Warshall paralelo con actores Ray (CPU)
│   ├── gpu/                 # Variantes GPU: básica (CuPy), segmentada, multi-GPU
│   └── utils/               # Monitor de recursos, métricas, exportador JSON
├── experimentos/
│   ├── config.py            # Parámetros globales del experimento
│   ├── escenarios.py        # Definición de los grupos E1–E5 y E_GPU*
│   └── ejecutar_benchmarks.py
├── analisis/
│   ├── generar_graficos.py  # Figuras PDF/PNG para el artículo
│   └── generar_tablas.py    # Tablas LaTeX
├── scripts/
│   ├── test_docker.sh       # Test de humo Docker
│   └── patagon/             # Scripts SLURM para el clúster Patagón
├── docker/
│   ├── Dockerfile
│   └── entrypoint.sh
├── .github/workflows/
│   └── docker.yml           # CI: build y publish en GHCR
├── docker-compose.yml
├── requirements.txt
├── Makefile
└── IEEEtran.cls
```

---

## Hardware utilizado (clúster Patagón, UACh)

| Componente | Especificación |
|---|---|
| CPU | AMD EPYC 9534 (64 cores) |
| RAM | 384 GB DDR5 |
| GPU | 3 × NVIDIA RTX A4000 16 GB |
| Red | InfiniBand HDR 200 Gb/s |
| Scheduler | SLURM 23.x |

---

## Comandos Make

```
make docker-build        Construir imagen Docker localmente
make docker-test         Test de humo (CPU, ~1 min)
make docker-shell        Shell interactivo (CPU)
make docker-shell-gpu    Shell interactivo (GPU)
make docker-benchmark    Benchmark CPU completo en Docker
make docker-benchmark-gpu  Benchmark GPU en Docker
make benchmark           Benchmark local (sin Docker)
make analisis            Generar gráficos y tablas
make limpiar             Eliminar resultados generados
make ayuda               Ver todos los comandos
```

---

## Reproducción de los experimentos del artículo

Los resultados del artículo se obtuvieron ejecutando los scripts SLURM
en el clúster Patagón de la UACh:

```bash
# GPU (3× A4000) — escenarios E_GPU*
sbatch scripts/patagon/submit_ai.sbatch

# CPU (32 cores) — escenarios E1–E5
sbatch scripts/patagon/submit_cpu.sbatch
```

Los archivos `resultados/raw_*.json` no se incluyen en el repositorio
por su tamaño. El archivo `resultados/resultados_agregados.json` con
los estadísticos finales está disponible en los
[releases del repositorio](https://github.com/mittods/Floyd-Warshall-Ray/releases).


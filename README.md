# Floyd-Warshall con Ray

Entorno experimental para evaluar la paralelización del algoritmo
Floyd-Warshall mediante el framework Ray sobre hardware HPC.

[![Build and Publish Docker](https://github.com/martinmaza/floyd-warshall-ray/actions/workflows/docker.yml/badge.svg)](https://github.com/martinmaza/floyd-warshall-ray/actions/workflows/docker.yml)
[![Docker Image](https://ghcr.io/martinmaza/floyd-warshall-ray:latest)](https://github.com/martinmaza/floyd-warshall-ray/pkgs/container/floyd-warshall-ray)

---

## Descripción

Este repositorio contiene el entorno experimental completo para el artículo
científico:

> **"Paralelización del Algoritmo Floyd-Warshall mediante Ray: Análisis de
> Rendimiento y Escalabilidad en Hardware de Alto Rendimiento"**

El objetivo es responder experimentalmente:

- ¿Cuánto acelera Ray la ejecución de Floyd-Warshall?
- ¿Cuál es el speedup y la eficiencia paralela obtenidos?
- ¿Cuál es el overhead real de Ray y desde qué tamaño es rentable?
- ¿Cómo escala con el número de workers y el tamaño de la matriz?
- ¿Cuál es el impacto en consumo de CPU, memoria y energía?

---

## Inicio rápido

### Con Docker (recomendado)

```bash
# 1. Clonar el repositorio
git clone https://github.com/martinmaza/floyd-warshall-ray.git
cd floyd-warshall-ray

# 2. Descargar la imagen publicada
docker pull ghcr.io/martinmaza/floyd-warshall-ray:latest

# 3. Ejecutar benchmarks completos
docker compose run --rm benchmark make benchmark

# 4. Generar tablas y gráficos
docker compose run --rm benchmark make analisis
```

### Sin Docker

```bash
git clone https://github.com/martinmaza/floyd-warshall-ray.git
cd floyd-warshall-ray
python -m venv .venv && source .venv/bin/activate
make instalar
make validar
make benchmark
```

---

## Estructura del proyecto

```
Floyd-Warshall-Ray/
├── src/
│   ├── secuencial/              # Implementación secuencial (línea base)
│   │   └── floyd_warshall_secuencial.py
│   ├── ray_parallel/            # Implementación paralela con actores Ray
│   │   └── floyd_warshall_ray.py
│   └── utils/                   # Métricas, monitor del sistema, exportador
│       ├── metricas.py
│       ├── monitor.py
│       └── exportador.py
├── experimentos/
│   ├── config.py                # Parámetros configurables del experimento
│   ├── escenarios.py            # Definición de los 5 grupos de escenarios
│   └── ejecutar_benchmarks.py  # Script principal de ejecución
├── analisis/
│   ├── generar_tablas.py        # Tablas LaTeX desde resultados
│   └── generar_graficos.py      # Gráficos PDF/PNG desde resultados
├── scripts/                     # Scripts de automatización
├── docker/                      # Dockerfile y entrypoint
├── .github/workflows/           # CI/CD: build y publish en GHCR
├── resultados/                  # Resultados JSON/CSV/Parquet (generados)
├── graficos/                    # Figuras PDF/PNG (generadas)
├── manual/                      # Manual de reproducción
├── ray-ejemplos/                # Ejemplos de clase (referencia)
├── ray.tex                      # Artículo científico IEEE
├── METODOLOGIA_RAZONADA.md      # Razonamiento metodológico detallado
├── docker-compose.yml
├── requirements.txt
└── Makefile
```

---

## Imagen Docker

La imagen se publica automáticamente en GHCR con cada push a `main`.

```bash
# Última versión
docker pull ghcr.io/martinmaza/floyd-warshall-ray:latest

# Versión específica del artículo
docker pull ghcr.io/martinmaza/floyd-warshall-ray:v1.0

# Commit específico (hash de 7 chars)
docker pull ghcr.io/martinmaza/floyd-warshall-ray:abc1234
```

### Autenticación en GHCR

```bash
echo $GITHUB_TOKEN | docker login ghcr.io -u $GITHUB_USER --password-stdin
```

---

## Comandos disponibles

```
make ayuda            Ver todos los comandos
make benchmark        Ejecutar todos los experimentos
make secuencial       Solo versión secuencial
make ray              Solo versión con Ray
make tablas           Generar tablas LaTeX
make graficos         Generar gráficos
make analisis         Tablas + gráficos
make latex            Compilar artículo PDF
make articulo         analisis + latex (pipeline completo)
make validar          Verificar entorno
make docker-build     Construir imagen localmente
make docker-pull      Descargar imagen de GHCR
make exportar         Comprimir resultados
make limpiar          Eliminar resultados generados
```

---

## Hardware utilizado en el artículo

| Componente | Especificación |
|---|---|
| CPU | AMD Ryzen Threadripper PRO 5975WX (32c/64t) |
| RAM | 128 GB DDR4-3200 MHz |
| SSD | 2 TB NVMe |
| GPU | NVIDIA TITAN V 12 GB HBM2 |
| OS | Linux (kernel 7.x) |

---

## Configuración del experimento

| Parámetro | Valores |
|---|---|
| Tamaños de matriz (n) | 64, 128, 256, 512, 1024, 2048, 4096 |
| Número de actores Ray | 1, 2, 4, 8, 16, 32 |
| Repeticiones | 10 (con test de Grubbs α=0.05) |
| IC | 95% (distribución t de Student) |
| Semilla base | 42 |
| Densidad de grafo | 0.7 |

---

## Documentación adicional

- [Manual de reproducción completo](manual/MANUAL_REPRODUCCION.md)
- [Razonamiento metodológico](METODOLOGIA_RAZONADA.md)
- [Ejemplos de Ray (referencia)](ray-ejemplos/)

---

## Cita

Si utiliza este entorno experimental en su investigación, por favor cite:

```bibtex
@inproceedings{autor2026fw,
  title     = {Paralelización del Algoritmo Floyd-Warshall mediante Ray:
               Análisis de Rendimiento y Escalabilidad en Hardware de Alto Rendimiento},
  author    = {[AUTORES]},
  booktitle = {[CONFERENCIA]},
  year      = {2026},
  note      = {Código disponible en \url{https://github.com/martinmaza/floyd-warshall-ray}}
}
```

---

## Licencia

MIT

# Manual de Reproducción

**Proyecto:** Paralelización de Floyd-Warshall con Ray  
**Versión del manual:** 1.0  
**Última actualización:** 2026-06-29

Este manual permite reproducir exactamente los experimentos descritos en el
artículo, desde cero, en cualquier sistema con el hardware especificado.

---

## Índice

1. [Requisitos](#1-requisitos)
2. [Instalación con Docker (recomendado)](#2-instalación-con-docker-recomendado)
3. [Instalación sin Docker](#3-instalación-sin-docker)
4. [Ejecución de los experimentos](#4-ejecución-de-los-experimentos)
5. [Análisis y generación de figuras](#5-análisis-y-generación-de-figuras)
6. [Compilación del artículo LaTeX](#6-compilación-del-artículo-latex)
7. [Descripción de los scripts](#7-descripción-de-los-scripts)
8. [Variables de entorno y configuración](#8-variables-de-entorno-y-configuración)
9. [Validación de correctitud](#9-validación-de-correctitud)
10. [Solución de errores frecuentes](#10-solución-de-errores-frecuentes)
11. [Trazabilidad: imagen Docker ↔ artículo](#11-trazabilidad-imagen-docker--artículo)

---

## 1. Requisitos

### Hardware

| Componente | Mínimo | Usado en el artículo |
|---|---|---|
| CPU | 4 núcleos x86-64 | AMD Ryzen Threadripper PRO 5975WX (32c/64t) |
| RAM | 16 GB | 128 GB DDR4-3200 |
| Almacenamiento | 10 GB libres | SSD NVMe 2 TB |
| GPU | Opcional | NVIDIA TITAN V 12 GB |

> Los resultados del artículo solo son reproducibles exactamente en el hardware
> idéntico. En hardware diferente se obtendrán tiempos distintos pero las
> tendencias (speedup relativo, punto de crossover) deben ser similares.

### Software

| Software | Versión mínima | Notas |
|---|---|---|
| Docker | 24.x | Para reproducción con contenedor |
| Docker Compose | 2.x | Incluido en Docker Desktop |
| Python | 3.10 | Solo si no usa Docker |
| Git | 2.x | Para clonar el repositorio |
| nvidia-container-toolkit | Opcional | Solo si usa GPU |

---

## 2. Instalación con Docker (recomendado)

Este es el método de reproducción garantizada. No requiere instalar Python
ni dependencias manualmente.

### 2.1 Autenticar en GitHub Container Registry

```bash
# Usando token personal de GitHub (PAT) con permiso read:packages
echo $GITHUB_TOKEN | docker login ghcr.io -u $GITHUB_USER --password-stdin

# O con GitHub CLI:
gh auth token | docker login ghcr.io -u $(gh api user --jq .login) --password-stdin
```

### 2.2 Descargar la imagen

```bash
# Última versión (rama main)
docker pull ghcr.io/martinmaza/floyd-warshall-ray:latest

# Versión específica usada en el artículo (recomendado para reproducción exacta)
docker pull ghcr.io/martinmaza/floyd-warshall-ray:v1.0

# Commit específico (máxima trazabilidad)
docker pull ghcr.io/martinmaza/floyd-warshall-ray:<HASH_COMMIT>
```

> El hash del commit exacto utilizado en el artículo se especifica en la
> sección de metadatos del paper y en el archivo `resultados/metadatos_experimento.json`.

### 2.3 Clonar el repositorio

```bash
git clone https://github.com/martinmaza/floyd-warshall-ray.git
cd floyd-warshall-ray
```

### 2.4 Ejecutar los benchmarks

```bash
# Ejecutar todo el pipeline experimental
docker compose run --rm benchmark make benchmark

# O equivalentemente con docker run:
docker run --rm \
  -v $(pwd)/resultados:/app/resultados \
  -v $(pwd)/graficos:/app/graficos \
  --ipc=host \
  ghcr.io/martinmaza/floyd-warshall-ray:latest \
  make benchmark
```

### 2.5 Actualizar la imagen

```bash
docker pull ghcr.io/martinmaza/floyd-warshall-ray:latest
docker compose run --rm benchmark make benchmark
```

### 2.6 Reconstruir la imagen localmente (si es necesario)

```bash
# Solo si necesita modificar el código o las dependencias
make docker-build

# Ejecutar con la imagen local:
docker run --rm \
  -v $(pwd)/resultados:/app/resultados \
  floyd-warshall-ray:local \
  make benchmark
```

---

## 3. Instalación sin Docker

Si prefiere ejecutar directamente en el sistema host (menor nivel de
reproducibilidad garantizada).

### 3.1 Requisitos previos

```bash
# Verificar Python 3.10+
python --version

# En sistemas Debian/Ubuntu:
sudo apt-get install python3.10 python3.10-venv python3.10-dev
```

### 3.2 Crear entorno virtual

```bash
git clone https://github.com/martinmaza/floyd-warshall-ray.git
cd floyd-warshall-ray

python3.10 -m venv .venv
source .venv/bin/activate
```

### 3.3 Instalar dependencias

```bash
make instalar
# O directamente:
pip install -r requirements.txt
```

### 3.4 Verificar instalación

```bash
make validar
```

---

## 4. Ejecución de los experimentos

### 4.1 Ejecución completa (recomendada)

Ejecuta todos los escenarios experimentales en el orden correcto:

```bash
make benchmark
# Equivalente a: ./scripts/ejecutar_todos.sh
```

Este comando:
1. Verifica las dependencias.
2. Registra los metadatos del sistema.
3. Ejecuta todos los escenarios (E1–E5).
4. Exporta los resultados a JSON y CSV.

**Tiempo estimado:** 2–8 horas dependiendo del hardware.

### 4.2 Ejecutar solo el secuencial

```bash
make secuencial
# O: ./scripts/ejecutar_secuencial.sh
```

### 4.3 Ejecutar solo la versión Ray

```bash
make ray
# O: ./scripts/ejecutar_ray.sh
```

### 4.4 Ejecutar un escenario específico

```bash
# Ver todos los escenarios disponibles sin ejecutar:
make mostrar-escenarios

# Ejecutar solo el grupo E1 con n=512 y 8 workers:
FW_TAMANOS=512 FW_WORKERS=8 FW_REPETICIONES=3 python -m experimentos.ejecutar_benchmarks \
  --escenarios E1_comparacion
```

### 4.5 Ejecución rápida para verificación

Para verificar que el entorno funciona antes del benchmark completo:

```bash
FW_TAMANOS=64,128 FW_WORKERS=2,4 FW_REPETICIONES=3 \
  python -m experimentos.ejecutar_benchmarks \
  --escenarios E1_comparacion
```

---

## 5. Análisis y generación de figuras

Después de ejecutar los benchmarks, generar tablas y gráficos:

```bash
# Generar tablas LaTeX (en resultados/*.tex)
make tablas

# Generar gráficos PDF y PNG (en graficos/)
make graficos

# Ambos:
make analisis
```

Los archivos generados son:

| Archivo | Descripción |
|---|---|
| `resultados/tabla_tiempos.tex` | Tiempos secuencial vs. Ray |
| `resultados/tabla_speedup.tex` | Speedup y eficiencia |
| `resultados/tabla_overhead.tex` | Descomposición del overhead |
| `resultados/tabla_recursos.tex` | CPU, RAM, energía |
| `graficos/speedup_vs_tamano.pdf` | Speedup vs. n |
| `graficos/speedup_vs_workers.pdf` | Escalabilidad fuerte |
| `graficos/tiempo_vs_tamano.pdf` | Tiempo de ejecución |
| `graficos/overhead_vs_tamano.pdf` | Overhead de Ray |
| `graficos/eficiencia_vs_workers.pdf` | Eficiencia paralela |
| `graficos/escalabilidad_debil.pdf` | Escalabilidad débil |
| `graficos/consumo_cpu_ram.pdf` | Recursos del sistema |

---

## 6. Compilación del artículo LaTeX

```bash
# Requiere latexmk y texlive instalados (incluidos en la imagen Docker)
make latex

# Todo el pipeline: benchmark + análisis + latex:
make articulo
```

El artículo en `ray.tex` usa `\input{resultados/tabla_*.tex}` para incluir
los resultados automáticamente. Las figuras se incluyen desde `graficos/`.

---

## 7. Descripción de los scripts

| Script | Descripción |
|---|---|
| `scripts/construir_docker.sh` | Construye la imagen Docker localmente |
| `scripts/iniciar_ray.sh` | Inicia el cluster Ray local (sin Docker) |
| `scripts/detener_ray.sh` | Detiene el cluster Ray |
| `scripts/ejecutar_secuencial.sh` | Ejecuta benchmarks secuenciales |
| `scripts/ejecutar_ray.sh` | Ejecuta benchmarks con Ray |
| `scripts/ejecutar_todos.sh` | Pipeline completo de benchmarks |
| `scripts/generar_tablas.sh` | Genera tablas LaTeX |
| `scripts/generar_graficos.sh` | Genera gráficos PDF/PNG |
| `scripts/exportar_resultados.sh` | Comprime resultados en tar.gz |
| `scripts/limpiar_resultados.sh` | Elimina resultados generados |
| `scripts/validar_entorno.sh` | Verifica el entorno antes de ejecutar |

---

## 8. Variables de entorno y configuración

Todas las variables tienen valores por defecto definidos en
`experimentos/config.py`. Se pueden sobreescribir:

| Variable | Por defecto | Descripción |
|---|---|---|
| `FW_TAMANOS` | `64,128,256,512,1024,2048` | Tamaños de matriz (lista CSV) |
| `FW_WORKERS` | `1,2,4,8,16,32` | Número de actores Ray (lista CSV) |
| `FW_REPETICIONES` | `10` | Repeticiones por configuración |
| `FW_SEMILLA` | `42` | Semilla base para reproducibilidad |
| `FW_DENSIDAD` | `0.7` | Densidad de aristas del grafo |
| `FW_INTERVALO_MONITOR` | `0.5` | Intervalo de muestreo del monitor (s) |
| `RAY_NUM_CPUS` | (autodetección) | Número de CPUs para Ray |
| `RAY_OBJECT_STORE_MEMORY` | (autodetección) | Memoria del object store de Ray |

**Ejemplo de configuración personalizada:**
```bash
export FW_TAMANOS="256,512,1024"
export FW_WORKERS="4,8,16"
export FW_REPETICIONES="5"
make benchmark
```

---

## 9. Validación de correctitud

El sistema verifica automáticamente la correctitud de la versión Ray
comparándola con la versión secuencial. Si se detecta una discrepancia:

```
[ERROR] FALLO DE CORRECTITUD en escenario E1_ray_n256_w8, rep 0
```

Para ejecutar la verificación manualmente:

```bash
python -c "
import sys, numpy as np
sys.path.insert(0, '.')
from src.secuencial.floyd_warshall_secuencial import floyd_warshall_secuencial, inicializar_matriz
from src.ray_parallel.floyd_warshall_ray import floyd_warshall_ray
import ray

ray.init(num_cpus=4, ignore_reinit_error=True)

n = 128
m = inicializar_matriz(n, semilla=42)
resultado_seq, _ = floyd_warshall_secuencial(m)
resultado_ray, _ = floyd_warshall_ray(m, num_actores=4, inicializar=False)

if np.allclose(resultado_seq, resultado_ray, equal_nan=False, rtol=1e-9):
    print('CORRECTITUD OK: ambas versiones producen el mismo resultado')
else:
    diff = np.abs(resultado_seq - resultado_ray)
    print(f'ERROR: diferencia máxima = {diff.max():.2e}')
ray.shutdown()
"
```

---

## 10. Solución de errores frecuentes

### Error: `ray.exceptions.RayActorError`

**Causa:** Falta de memoria para los actores.  
**Solución:**
```bash
# Reducir el tamaño de la matriz o el número de actores:
FW_TAMANOS=512,1024 FW_WORKERS=4,8 make benchmark

# O aumentar la memoria del object store de Ray:
export RAY_OBJECT_STORE_MEMORY=10000000000  # 10 GB
```

### Error: `ImportError: No module named 'pynvml'`

**Causa:** pynvml no está instalado (solo afecta métricas GPU).  
**Solución:**
```bash
pip install pynvml
# El benchmark continúa sin métricas GPU si no está disponible.
```

### Error: `Permission denied` al leer `/sys/class/powercap/`

**Causa:** Falta de permisos para métricas de energía.  
**Solución:**
```bash
sudo chmod a+r /sys/class/powercap/*/energy_uj
# O ejecutar el benchmark como root (no recomendado para producción).
# Las métricas de energía quedarán en 0.0 si no hay acceso.
```

### Error: `latexmk: command not found`

**Causa:** LaTeX no está instalado.  
**Solución:**
```bash
# En Debian/Ubuntu:
sudo apt-get install texlive-latex-extra latexmk

# Con Docker (ya incluido en la imagen):
docker compose run --rm benchmark make latex
```

### Error: `docker: Cannot connect to the Docker daemon`

**Causa:** Docker no está corriendo.  
**Solución:**
```bash
sudo systemctl start docker
# O en macOS/Windows: iniciar Docker Desktop
```

### Error: `ModuleNotFoundError: No module named 'src'`

**Causa:** El PYTHONPATH no incluye el directorio raíz del proyecto.  
**Solución:**
```bash
export PYTHONPATH=$(pwd):$PYTHONPATH
# O usar make que lo configura automáticamente:
make benchmark
```

### Imagen GPU no detectada en Docker

**Causa:** nvidia-container-toolkit no está instalado.  
**Solución:**
```bash
# Instalar nvidia-container-toolkit:
sudo apt-get install nvidia-container-toolkit
sudo systemctl restart docker
# Luego descomentar la sección de GPU en docker-compose.yml
```

---

## 11. Trazabilidad: imagen Docker ↔ artículo

Para garantizar que los resultados del artículo son reproducibles, toda
publicación debe registrar:

```
Entorno experimental:
  - Repositorio: https://github.com/martinmaza/floyd-warshall-ray
  - Commit Git: <HASH_COMPLETO_40_CHARS>
  - Tag Git: v1.0
  - Imagen Docker: ghcr.io/martinmaza/floyd-warshall-ray:<HASH_COMMIT_7_CHARS>
  - Fecha de construcción: 2026-XX-XX
```

Esta información se almacena automáticamente en
`resultados/metadatos_experimento.json` durante la ejecución.

Para reproducir exactamente los resultados del artículo:

```bash
# 1. Usar el commit exacto:
git checkout <HASH_COMMIT>

# 2. Usar la imagen exacta:
docker pull ghcr.io/martinmaza/floyd-warshall-ray:<HASH_COMMIT_7_CHARS>

# 3. Ejecutar:
docker run --rm \
  -v $(pwd)/resultados:/app/resultados \
  -v $(pwd)/graficos:/app/graficos \
  ghcr.io/martinmaza/floyd-warshall-ray:<HASH_COMMIT_7_CHARS> \
  make benchmark
```

---

*Fin del manual de reproducción.*

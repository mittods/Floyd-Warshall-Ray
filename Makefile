# Makefile — Automatización del entorno experimental Floyd-Warshall con Ray
# Requiere: GNU make 4.x, Python 3.10+, Ray 2.44+
# Documentación completa: manual/MANUAL_REPRODUCCION.md

SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c
.DEFAULT_GOAL := ayuda

PYTHON := python
PROJECT_DIR := $(shell pwd)
PYTHONPATH := $(PROJECT_DIR)

export PYTHONPATH

# ── Colores para salida legible ──────────────────────────────────────────────
BOLD  := \033[1m
RESET := \033[0m
GREEN := \033[32m
CYAN  := \033[36m
YELLOW := \033[33m

# ── Ayuda ────────────────────────────────────────────────────────────────────
.PHONY: ayuda
ayuda:
	@echo ""
	@echo "$(BOLD)Floyd-Warshall con Ray — Comandos disponibles$(RESET)"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo ""
	@echo "$(BOLD)CONFIGURACIÓN$(RESET)"
	@echo "  $(CYAN)make instalar$(RESET)          Instalar dependencias Python"
	@echo "  $(CYAN)make validar$(RESET)           Validar entorno antes de ejecutar"
	@echo ""
	@echo "$(BOLD)DOCKER$(RESET)"
	@echo "  $(CYAN)make docker-build$(RESET)          Construir imagen Docker localmente"
	@echo "  $(CYAN)make docker-pull$(RESET)           Descargar imagen desde GHCR"
	@echo "  $(CYAN)make docker-shell$(RESET)          Shell interactivo (CPU)"
	@echo "  $(CYAN)make docker-shell-gpu$(RESET)      Shell interactivo (GPU, requiere nvidia-container-toolkit)"
	@echo "  $(CYAN)make docker-benchmark$(RESET)      Benchmarks CPU en Docker"
	@echo "  $(CYAN)make docker-benchmark-gpu$(RESET)  Benchmarks GPU en Docker"
	@echo ""
	@echo "$(BOLD)EJECUCIÓN$(RESET)"
	@echo "  $(CYAN)make benchmark$(RESET)         Ejecutar todos los experimentos (recomendado)"
	@echo "  $(CYAN)make secuencial$(RESET)        Ejecutar solo versión secuencial"
	@echo "  $(CYAN)make ray$(RESET)               Ejecutar solo versión Ray"
	@echo "  $(CYAN)make mostrar-escenarios$(RESET) Listar todos los escenarios sin ejecutar"
	@echo ""
	@echo "$(BOLD)ANÁLISIS$(RESET)"
	@echo "  $(CYAN)make tablas$(RESET)            Generar tablas LaTeX"
	@echo "  $(CYAN)make graficos$(RESET)          Generar gráficos PDF/PNG"
	@echo "  $(CYAN)make analisis$(RESET)          Generar tablas + gráficos"
	@echo "  $(CYAN)make latex$(RESET)             Compilar artículo ray.tex → ray.pdf"
	@echo "  $(CYAN)make articulo$(RESET)          analisis + latex"
	@echo ""
	@echo "$(BOLD)MANTENIMIENTO$(RESET)"
	@echo "  $(CYAN)make exportar$(RESET)          Empaquetar resultados en tar.gz"
	@echo "  $(CYAN)make limpiar$(RESET)           Eliminar resultados generados"
	@echo "  $(CYAN)make limpiar-todo$(RESET)      Eliminar resultados y caché Python"
	@echo ""

# ── Instalación ──────────────────────────────────────────────────────────────
.PHONY: instalar
instalar:
	@echo "$(BOLD)Instalando dependencias...$(RESET)"
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt
	@echo "$(GREEN)Instalación completada.$(RESET)"

# ── Validación ───────────────────────────────────────────────────────────────
.PHONY: validar
validar:
	@echo "$(BOLD)Validando entorno...$(RESET)"
	@bash scripts/validar_entorno.sh

# ── Docker ───────────────────────────────────────────────────────────────────
.PHONY: docker-build
docker-build:
	@echo "$(BOLD)Construyendo imagen CPU (sin CUDA)...$(RESET)"
	docker build --target cpu -t floyd-warshall-ray:cpu -f docker/Dockerfile .
	@echo "$(GREEN)Imagen floyd-warshall-ray:cpu lista.$(RESET)"

.PHONY: docker-build-gpu
docker-build-gpu:
	@echo "$(BOLD)Construyendo imagen GPU (con CuPy/CUDA, tarda ~10 min)...$(RESET)"
	docker build --target gpu -t floyd-warshall-ray:gpu -f docker/Dockerfile .

.PHONY: docker-pull
docker-pull:
	@echo "$(BOLD)Descargando imagen desde GHCR...$(RESET)"
	docker pull ghcr.io/mittods/floyd-warshall-ray:latest

.PHONY: docker-shell
docker-shell:
	@echo "$(BOLD)Iniciando shell interactivo en Docker (CPU)...$(RESET)"
	docker compose run --rm benchmark bash

.PHONY: docker-shell-gpu
docker-shell-gpu:
	@echo "$(BOLD)Iniciando shell interactivo en Docker (GPU, requiere nvidia-container-toolkit)...$(RESET)"
	docker compose --profile gpu run --rm benchmark-gpu bash

.PHONY: docker-benchmark
docker-benchmark:
	@echo "$(BOLD)Ejecutando benchmarks CPU en Docker...$(RESET)"
	docker compose run --rm benchmark make benchmark

.PHONY: docker-benchmark-gpu
docker-benchmark-gpu:
	@echo "$(BOLD)Ejecutando benchmarks GPU en Docker (requiere nvidia-container-toolkit)...$(RESET)"
	docker compose --profile gpu run --rm benchmark-gpu make benchmark

.PHONY: docker-test
docker-test:
	@echo "$(BOLD)Ejecutando test de humo Docker...$(RESET)"
	@bash scripts/test_docker.sh floyd-warshall-ray:local

# ── Ejecución de benchmarks ──────────────────────────────────────────────────
.PHONY: benchmark
benchmark:
	@echo "$(BOLD)Ejecutando todos los benchmarks...$(RESET)"
	@bash scripts/ejecutar_todos.sh

.PHONY: secuencial
secuencial:
	@echo "$(BOLD)Ejecutando benchmarks secuenciales...$(RESET)"
	@bash scripts/ejecutar_secuencial.sh

.PHONY: ray
ray:
	@echo "$(BOLD)Ejecutando benchmarks con Ray...$(RESET)"
	@bash scripts/ejecutar_ray.sh

.PHONY: mostrar-escenarios
mostrar-escenarios:
	@echo "$(BOLD)Escenarios experimentales definidos:$(RESET)"
	$(PYTHON) -m experimentos.ejecutar_benchmarks --solo-mostrar

# ── Análisis y visualización ─────────────────────────────────────────────────
.PHONY: tablas
tablas:
	@echo "$(BOLD)Generando tablas LaTeX...$(RESET)"
	@bash scripts/generar_tablas.sh

.PHONY: graficos
graficos:
	@echo "$(BOLD)Generando gráficos...$(RESET)"
	@bash scripts/generar_graficos.sh

.PHONY: analisis
analisis: tablas graficos
	@echo "$(GREEN)Análisis completado. Ver resultados/ y graficos/$(RESET)"

.PHONY: latex
latex:
	@echo "$(BOLD)Compilando artículo LaTeX...$(RESET)"
	latexmk -pdf -interaction=nonstopmode ray.tex
	@echo "$(GREEN)Artículo compilado: ray.pdf$(RESET)"

.PHONY: articulo
articulo: analisis latex

# ── Ray (cluster local) ──────────────────────────────────────────────────────
.PHONY: ray-iniciar
ray-iniciar:
	@bash scripts/iniciar_ray.sh

.PHONY: ray-detener
ray-detener:
	@bash scripts/detener_ray.sh

# ── Exportación y limpieza ───────────────────────────────────────────────────
.PHONY: exportar
exportar:
	@bash scripts/exportar_resultados.sh

.PHONY: limpiar
limpiar:
	@bash scripts/limpiar_resultados.sh

.PHONY: limpiar-todo
limpiar-todo: limpiar
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -name "*.pyc" -delete 2>/dev/null || true
	@find . -name "*.pyo" -delete 2>/dev/null || true
	@rm -f ray.pdf ray.log ray.aux ray.bbl ray.blg ray.fls ray.fdb_latexmk
	@echo "$(GREEN)Limpieza completa realizada.$(RESET)"

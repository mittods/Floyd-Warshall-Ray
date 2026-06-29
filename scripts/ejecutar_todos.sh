#!/bin/bash
# Ejecuta el ciclo completo de benchmarks, análisis y generación de figuras.
#
# Este script orquesta:
#   1. Verificación del entorno
#   2. Ejecución de todos los escenarios experimentales
#   3. Generación de tablas LaTeX
#   4. Generación de gráficos
#   5. Compilación del artículo LaTeX (opcional)
#
# Uso: ./scripts/ejecutar_todos.sh [--compilar-latex]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROYECTO_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROYECTO_DIR"

export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

COMPILAR_LATEX=false
for arg in "$@"; do
    [[ "$arg" == "--compilar-latex" ]] && COMPILAR_LATEX=true
done

echo "══════════════════════════════════════════════════════════════════════"
echo "  Floyd-Warshall con Ray — Benchmarks Completos"
echo "══════════════════════════════════════════════════════════════════════"
echo ""
echo "Configuración del experimento:"
echo "  Tamaños de matriz : ${FW_TAMANOS:-64,128,256,512,1024,2048}"
echo "  Workers Ray       : ${FW_WORKERS:-1,2,4,8,16,32}"
echo "  Repeticiones      : ${FW_REPETICIONES:-10}"
echo "  Semilla           : ${FW_SEMILLA:-42}"
echo "  Densidad          : ${FW_DENSIDAD:-0.7}"
echo ""

# Verificar dependencias
echo "── Verificando dependencias ──────────────────────────────────────────"
python -c "
import ray, numpy, scipy, pandas, matplotlib, psutil
print(f'Ray: {ray.__version__}, NumPy: {numpy.__version__}')
print(f'SciPy: {scipy.__version__}, pandas: {pandas.__version__}')
print(f'matplotlib: {matplotlib.__version__}, psutil: {psutil.__version__}')
"
echo ""

# Crear directorios de salida
mkdir -p resultados graficos metricas

# Registrar metadatos del experimento
TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "sin-git")
cat > resultados/metadatos_experimento.json << EOF
{
  "timestamp": "$TIMESTAMP",
  "git_commit": "$COMMIT",
  "host": "$(hostname)",
  "cpu": "$(grep 'model name' /proc/cpuinfo | head -1 | cut -d: -f2 | xargs)",
  "num_cpus": $(nproc),
  "ram_gb": $(free -g | awk '/^Mem:/{print $2}'),
  "fw_tamanos": "${FW_TAMANOS:-64,128,256,512,1024,2048}",
  "fw_workers": "${FW_WORKERS:-1,2,4,8,16,32}",
  "fw_repeticiones": ${FW_REPETICIONES:-10},
  "fw_semilla": ${FW_SEMILLA:-42}
}
EOF

echo "── Ejecutando benchmarks ─────────────────────────────────────────────"
T_INICIO=$SECONDS
python -m experimentos.ejecutar_benchmarks
T_FIN=$SECONDS

echo ""
echo "Benchmarks completados en $((T_FIN - T_INICIO)) segundos."
echo ""

echo "── Generando tablas LaTeX ────────────────────────────────────────────"
python -m analisis.generar_tablas
echo ""

echo "── Generando gráficos ────────────────────────────────────────────────"
python -m analisis.generar_graficos
echo ""

if [ "$COMPILAR_LATEX" = true ]; then
    echo "── Compilando artículo LaTeX ─────────────────────────────────────────"
    latexmk -pdf -interaction=nonstopmode ray.tex
    echo "Artículo compilado: ray.pdf"
    echo ""
fi

echo "══════════════════════════════════════════════════════════════════════"
echo "  Proceso completado."
echo "  Resultados : resultados/"
echo "  Gráficos   : graficos/"
echo "══════════════════════════════════════════════════════════════════════"

#!/bin/bash
# Test de humo para la imagen Docker de Floyd-Warshall con Ray.
#
# Uso:
#   bash scripts/test_docker.sh                         # usa imagen local (build previo)
#   bash scripts/test_docker.sh ghcr.io/mittods/floyd-warshall-ray:latest
#
# Qué verifica:
#   1. La imagen arranca y las dependencias están presentes.
#   2. El benchmark CPU mínimo (n=128, E1+E2, 1 repetición) termina sin error.
#   3. Se genera resultados_agregados.json con al menos un registro.

set -euo pipefail

IMAGE="${1:-floyd-warshall-ray:cpu}"
TMPDIR="$(mktemp -d)"
trap 'rm -rf "$TMPDIR"' EXIT

echo "=== Test de humo: Floyd-Warshall con Ray ==="
echo "Imagen : $IMAGE"
echo "Salida : $TMPDIR/resultados/"
echo ""

# ── 1. Verificar dependencias ─────────────────────────────────────────────────
echo "[1/3] Verificando dependencias..."
docker run --rm "$IMAGE" python -c "
import ray, numpy, scipy, pandas, matplotlib, psutil
print(f'  Ray        {ray.__version__}')
print(f'  NumPy      {numpy.__version__}')
print(f'  Pandas     {pandas.__version__}')
print(f'  Matplotlib {matplotlib.__version__}')
print(f'  psutil     {psutil.__version__}')
"
echo "      OK"

# ── 2. Ejecutar benchmark CPU mínimo ─────────────────────────────────────────
echo "[2/3] Ejecutando benchmark CPU mínimo (n=128, 1 repetición)..."
mkdir -p "$TMPDIR/resultados"

docker run --rm \
  -v "$TMPDIR/resultados:/app/resultados" \
  -e FW_TAMANOS=128 \
  -e FW_WORKERS="1,2" \
  -e FW_REPETICIONES=1 \
  "$IMAGE" \
  python -m experimentos.ejecutar_benchmarks \
    --escenarios E1_comparacion E2_escal_fuerte

echo "      OK"

# ── 3. Verificar salida ───────────────────────────────────────────────────────
echo "[3/3] Verificando resultados_agregados.json..."
ARCHIVO="$TMPDIR/resultados/resultados_agregados.json"

if [ ! -f "$ARCHIVO" ]; then
  echo "FALLO: no se generó $ARCHIVO"
  exit 1
fi

N=$(python3 -c "import json; d=json.load(open('$ARCHIVO')); print(len(d))")
if [ "$N" -lt 1 ]; then
  echo "FALLO: resultados_agregados.json está vacío"
  exit 1
fi

echo "      OK ($N registros generados)"
echo ""
echo "=== Test PASADO ==="

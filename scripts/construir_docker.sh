#!/bin/bash
# Construye la imagen Docker localmente desde el Dockerfile del proyecto.
# No es necesario ejecutar esto si se utiliza la imagen publicada en GHCR.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROYECTO_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROYECTO_DIR"

TAG="${1:-floyd-warshall-ray:local}"
COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "sin-git")
FECHA=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

echo "Construyendo imagen Docker: $TAG"
echo "Commit: $COMMIT | Fecha: $FECHA"
echo ""

docker build \
    --file docker/Dockerfile \
    --tag "$TAG" \
    --build-arg GIT_COMMIT="$COMMIT" \
    --build-arg BUILD_DATE="$FECHA" \
    --label "git.commit=$COMMIT" \
    --label "build.date=$FECHA" \
    --progress=plain \
    .

echo ""
echo "Imagen construida exitosamente: $TAG"
echo ""
echo "Para ejecutar:"
echo "  docker run --rm -v \$(pwd)/resultados:/app/resultados $TAG bash"

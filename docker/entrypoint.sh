#!/bin/bash
# Entrypoint del contenedor Floyd-Warshall con Ray
# Configura el entorno y ejecuta el comando recibido

set -e

# Verificar que Python esté disponible
python --version

# Verificar que Ray esté instalado
python -c "import ray; print(f'Ray versión: {ray.__version__}')"

# Verificar que las dependencias principales estén disponibles
python -c "
import numpy, scipy, pandas, matplotlib, psutil
print(f'NumPy: {numpy.__version__}')
print(f'SciPy: {scipy.__version__}')
print(f'Pandas: {pandas.__version__}')
print(f'Matplotlib: {matplotlib.__version__}')
print(f'psutil: {psutil.__version__}')
"

# Informar si NVIDIA GPU está disponible
if command -v nvidia-smi &>/dev/null; then
    echo "GPU disponible:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "Advertencia: nvidia-smi no disponible (sin GPU o sin driver)"
fi

echo ""
echo "Entorno listo. Ejecutando: $*"
echo ""

exec "$@"

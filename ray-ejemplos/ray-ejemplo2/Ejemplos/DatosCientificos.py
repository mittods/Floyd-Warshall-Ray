import ray
import time
import numpy as np
import math
import random
import hashlib

# ========== INICIALIZAR RAY ==========
print("🚀 Inicializando Ray")
try:
    ray.init(include_dashboard=False, ignore_reinit_error=True)
    num_cpus = int(ray.cluster_resources().get('CPU', 4))
    print(f"✅ Ray inicializado con {num_cpus} CPUs disponibles")
except:
    ray.init(local_mode=True, ignore_reinit_error=True)
    num_cpus = 1
    print("✅ Ray inicializado en modo local")

print(f"💪 Preparándose para computación usando {num_cpus} cores...")

#  CONFIGURACION DE PRUEBA

tamano_dataset = 100_000  # 100k puntos de datos (Aumentar para mas carga)
iteraciones_por_chunk = 2000  # Análisis muy intensivo (Aumentar para mas carga)


# ========== EJEMPLO 3: SIMULACIÓN CIENTÍFICA MASIVA ==========
print("\n" + "="*70)
print("🧬 EJEMPLO 3: SIMULACIÓN CIENTÍFICA - ANÁLISIS DE DATOS MASIVOS")
print("="*70)

def analizar_dataset_complejo(datos, iteraciones=1000):
    """Análisis estadístico complejo de un dataset"""
    n = len(datos)
    resultados = {}
    
    # Múltiples análisis estadísticos complejos
    for _ in range(iteraciones):
        # Transformaciones matemáticas complejas
        transformados = np.array([
            np.sin(x) * np.cos(x**2) + np.exp(-x/1000) * np.log(abs(x) + 1)
            for x in datos
        ])
        
        # Cálculos estadísticos
        media = np.mean(transformados)
        std = np.std(transformados)
        skewness = np.mean(((transformados - media) / std) ** 3)
        kurtosis = np.mean(((transformados - media) / std) ** 4) - 3
        
        # Correlaciones cruzadas
        correlacion = np.corrcoef(datos, transformados)[0, 1]
        
    resultados = {
        'media_final': media,
        'std_final': std,
        'skewness': skewness,
        'kurtosis': kurtosis,
        'correlacion': correlacion,
        'n_iteraciones': iteraciones
    }
    
    return resultados

@ray.remote
def procesar_dataset_chunk(chunk_datos, chunk_id, iteraciones=1000):
    """Procesa un chunk del dataset con análisis complejo"""
    print(f"    Procesando chunk {chunk_id}...")
    return analizar_dataset_complejo(chunk_datos, iteraciones)

# Generar dataset masivo
print("Generando dataset científico masivo...")
#tamano_dataset = 100_000  # 100k puntos de datos
num_chunks_data = num_cpus * 4  # Más chunks para mejor paralelización
#iteraciones_por_chunk = 2000  # Análisis muy intensivo

# Dataset con distribución compleja
np.random.seed(42)
dataset_completo = np.concatenate([
    np.random.normal(0, 1, tamano_dataset // 3),
    np.random.exponential(2, tamano_dataset // 3),
    np.random.gamma(2, 2, tamano_dataset - 2*(tamano_dataset // 3))
])

# Dividir en chunks
chunk_size_data = len(dataset_completo) // num_chunks_data
chunks_data = [
    dataset_completo[i:i + chunk_size_data] 
    for i in range(0, len(dataset_completo), chunk_size_data)
]

print(f"🧬 Analizando dataset de {tamano_dataset:,} puntos")
print(f"📊 {num_chunks_data} chunks con {iteraciones_por_chunk} iteraciones cada uno")
print(f"⚡ Total de operaciones: {tamano_dataset * iteraciones_por_chunk:,}")

# SECUENCIAL
print("\n🐌 Análisis secuencial...")
inicio_seq = time.time()
resultados_seq_data = []
for i, chunk in enumerate(chunks_data):
    print(f"  Chunk secuencial {i+1}/{len(chunks_data)}...")
    resultado = procesar_dataset_chunk.remote.__wrapped__(chunk, i, iteraciones_por_chunk)
    resultados_seq_data.append(resultado)
tiempo_seq_data = time.time() - inicio_seq

# PARALELO
print("\n🚀 Análisis paralelo con Ray...")
inicio_ray = time.time()
futures_data = [
    procesar_dataset_chunk.remote(chunk, i, iteraciones_por_chunk) 
    for i, chunk in enumerate(chunks_data)
]

# Monitorear progreso científico
completed_science = 0
resultados_ray_data = []
while futures_data:
    ready, futures_data = ray.wait(futures_data, num_returns=1, timeout=10)
    for future in ready:
        resultado = ray.get(future)
        resultados_ray_data.append(resultado)
        completed_science += 1
        print(f"  Análisis científico {completed_science}/{num_chunks_data} completado...")

tiempo_ray_data = time.time() - inicio_ray

print(f"\n📊 RESULTADOS - ANÁLISIS CIENTÍFICO MASIVO:")
print(f"Puntos de datos: {tamano_dataset:,}")
print(f"Iteraciones por chunk: {iteraciones_por_chunk:,}")
print(f"Operaciones totales: {tamano_dataset * iteraciones_por_chunk:,}")
print(f"Tiempo secuencial: {tiempo_seq_data:.1f}s ({tiempo_seq_data/60:.1f} min)")
print(f"Tiempo Ray: {tiempo_ray_data:.1f}s ({tiempo_ray_data/60:.1f} min)")
print(f"Speedup: {tiempo_seq_data/tiempo_ray_data:.2f}x más rápido")
print(f"Throughput secuencial: {(tamano_dataset * iteraciones_por_chunk)/tiempo_seq_data:,.0f} ops/seg")
print(f"Throughput Ray: {(tamano_dataset * iteraciones_por_chunk)/tiempo_ray_data:,.0f} ops/seg")

# Mostrar algunos resultados científicos
if resultados_ray_data:
    media_general = np.mean([r['media_final'] for r in resultados_ray_data])
    std_general = np.mean([r['std_final'] for r in resultados_ray_data])
    print(f"📈 Media general del análisis: {media_general:.6f}")
    print(f"📊 Desviación estándar promedio: {std_general:.6f}")

import ray
import time
import numpy as np
import math
import random
import hashlib

# ========== INICIALIZAR RAY ==========
print("🚀 Inicializando Ray para computación EXTREMA...")
try:
    ray.init(include_dashboard=False, ignore_reinit_error=True)
    num_cpus = int(ray.cluster_resources().get('CPU', 4))
    print(f"✅ Ray inicializado con {num_cpus} CPUs disponibles")
except:
    ray.init(local_mode=True, ignore_reinit_error=True)
    num_cpus = 1
    print("✅ Ray inicializado en modo local")

print(f"💪 Preparándose para computación INTENSIVA usando {num_cpus} cores...")
print("⚠️  Este demo puede tomar varios minutos y usar 100% CPU")

# CONFIGURACIÓN DE PRUEBAS

# Ejemplo 2: DIFICULTAD DE HASH MINING
dificultad = 5  # Cantidad de 0 iniciales en el hash | Default: 5 (mayor a 5 es mas complicado)

#Ejemplo 3: ANÁLISIS CIENTÍFICO MASIVO
tamano_dataset = 100_000  # 100k puntos de datos (Aumentar para mas carga)
iteraciones_por_chunk = 2000  # Análisis muy intensivo (Aumentar para mas carga)

# ========== EJEMPLO 1: FACTORIZACIÓN DE NÚMEROS GRANDES ==========
print("\n" + "="*70)
print("🔥 EJEMPLO 1: FACTORIZACIÓN DE NÚMEROS SEMI-PRIMOS GRANDES")
print("="*70)

def factorizar_numero(n):
    """Factoriza un número usando fuerza bruta (MUY CPU-intensivo)"""
    if n <= 1:
        return []
    
    factores = []
    d = 2
    original_n = n
    
    # Factorización por fuerza bruta
    while d * d <= n:
        while n % d == 0:
            factores.append(d)
            n //= d
        d += 1
    
    if n > 1:
        factores.append(n)
    
    return original_n, factores

@ray.remote
def factorizar_rango_numeros(numeros):
    """Factoriza múltiples números grandes"""
    resultados = []
    for num in numeros:
        resultado = factorizar_numero(num)
        resultados.append(resultado)
    return resultados

# Generar números semi-primos grandes (producto de dos primos)
print("Generando números semi-primos EXTREMADAMENTE grandes...")
numeros_grandes = [
    982451653 * 982451681,    # ~9.6 × 10^17
    982451701 * 982451707,    # ~9.6 × 10^17  
    982451719 * 982451729,    # ~9.6 × 10^17
    982451749 * 982451753,    # ~9.6 × 10^17
    982451761 * 982451767,    # ~9.6 × 10^17
    982451783 * 982451797,    # ~9.6 × 10^17
    982451801 * 982451821,    # ~9.6 × 10^17
    982451831 * 982451837,    # ~9.6 × 10^17
    982451861 * 982451879,    # ~9.6 × 10^17
    982451881 * 982451893,    # ~9.6 × 10^17
    982451909 * 982451926,    # ~9.6 × 10^17
    982451939 * 982451953,    # ~9.6 × 10^17
]

print(f"📊 Factorizando {len(numeros_grandes)} números de ~18 dígitos cada uno")
print("⏱️  Esto puede tomar 2-5 minutos dependiendo de tu CPU...")

# Dividir trabajo entre CPUs disponibles
chunk_size = max(1, len(numeros_grandes) // num_cpus)
chunks = [numeros_grandes[i:i + chunk_size] for i in range(0, len(numeros_grandes), chunk_size)]

# EJECUCIÓN SECUENCIAL
print("\n🐌 Factorización secuencial (prepárate para esperar...)...")
inicio_seq = time.time()
resultados_seq = []
for i, chunk in enumerate(chunks):
    print(f"  Procesando chunk {i+1}/{len(chunks)}...")
    chunk_result = factorizar_rango_numeros.remote.__wrapped__(chunk)
    resultados_seq.extend(chunk_result)
tiempo_seq_fact = time.time() - inicio_seq

# EJECUCIÓN PARALELA
print("\n🚀 Factorización paralela con Ray...")
inicio_ray = time.time()
futures = [factorizar_rango_numeros.remote(chunk) for chunk in chunks]

# Monitorear progreso
completed = 0
while futures:
    ready, futures = ray.wait(futures, num_returns=1, timeout=10)
    if ready:
        completed += 1
        print(f"  Chunk {completed}/{len(chunks)} completado...")

resultados_ray = ray.get([factorizar_rango_numeros.remote(chunk) for chunk in chunks])
resultados_ray_flat = [item for sublist in resultados_ray for item in sublist]
tiempo_ray_fact = time.time() - inicio_ray

print(f"\n📊 RESULTADOS - FACTORIZACIÓN EXTREMA:")
print(f"Números factorizados: {len(numeros_grandes)}")
print(f"Dígitos por número: ~18")
print(f"Tiempo secuencial: {tiempo_seq_fact:.1f}s ({tiempo_seq_fact/60:.1f} min)")
print(f"Tiempo Ray: {tiempo_ray_fact:.1f}s ({tiempo_ray_fact/60:.1f} min)")
print(f"Speedup: {tiempo_seq_fact/tiempo_ray_fact:.2f}x más rápido")
print(f"Tiempo ahorrado: {(tiempo_seq_fact-tiempo_ray_fact)/60:.1f} minutos")

# ========== EJEMPLO 2: CRIPTOGRAFÍA - HASH MINING ==========
print("\n" + "="*70)
print("⛏️  EJEMPLO 2: HASH MINING (Simulación de Minería de Criptomonedas)")
print("="*70)

def minar_hash_secuencial(prefijo, dificultad, max_nonce):
    """Encuentra un hash que empiece con cierto número de ceros"""
    encontrados = []
    for nonce in range(max_nonce):
        data = f"{prefijo}{nonce}".encode()
        hash_result = hashlib.sha256(data).hexdigest()
        if hash_result.startswith('0' * dificultad):
            encontrados.append((nonce, hash_result))
    return encontrados

@ray.remote
def minar_hash_chunk(prefijo, dificultad, start_nonce, end_nonce):
    """Mina hashes en un rango específico"""
    encontrados = []
    for nonce in range(start_nonce, end_nonce):
        data = f"{prefijo}{nonce}".encode()
        hash_result = hashlib.sha256(data).hexdigest()
        if hash_result.startswith('0' * dificultad):
            encontrados.append((nonce, hash_result))
    return encontrados

# Configuración de minería
prefijo = "RayDemo_Block_"
#dificultad = 5  # Hash debe empezar con 5 ceros
total_nonces = 10_000_000  # 10 millones de intentos
chunk_size = total_nonces // (num_cpus * 2)  # Más chunks para mejor distribución

print(f"⛏️  Minando hashes con dificultad {dificultad} (prefijo: {'0' * dificultad})")
print(f"🔍 Probando {total_nonces:,} nonces...")
print(f"💎 Probabilidad de encontrar hash válido: ~{(1/16**dificultad)*total_nonces:.1f}")

# SECUENCIAL
print("\n🐌 Minería secuencial...")
inicio_seq = time.time()
hashes_encontrados_seq = minar_hash_secuencial(prefijo, dificultad, total_nonces)
tiempo_seq_hash = time.time() - inicio_seq

# PARALELO
print("\n🚀 Minería paralela con Ray...")
inicio_ray = time.time()
ranges = [(i, min(i + chunk_size, total_nonces)) for i in range(0, total_nonces, chunk_size)]
futures = [minar_hash_chunk.remote(prefijo, dificultad, start, end) for start, end in ranges]

# Monitorear progreso de minería
completed_mining = 0
while futures:
    ready, futures = ray.wait(futures, num_returns=1, timeout=5)
    if ready:
        completed_mining += 1
        print(f"  Chunk de minería {completed_mining}/{len(ranges)} completado...")

hashes_results = ray.get([minar_hash_chunk.remote(prefijo, dificultad, start, end) for start, end in ranges])
hashes_encontrados_ray = [hash_info for sublist in hashes_results for hash_info in sublist]
tiempo_ray_hash = time.time() - inicio_ray

print(f"\n📊 RESULTADOS - HASH MINING:")
print(f"Nonces probados: {total_nonces:,}")
print(f"Dificultad: {dificultad} ceros")
print(f"Hashes válidos encontrados (secuencial): {len(hashes_encontrados_seq)}")
print(f"Hashes válidos encontrados (Ray): {len(hashes_encontrados_ray)}")
print(f"Tiempo secuencial: {tiempo_seq_hash:.1f}s")
print(f"Tiempo Ray: {tiempo_ray_hash:.1f}s")
print(f"Speedup: {tiempo_seq_hash/tiempo_ray_hash:.2f}x más rápido")
print(f"Hash rate secuencial: {total_nonces/tiempo_seq_hash:,.0f} hashes/seg")
print(f"Hash rate Ray: {total_nonces/tiempo_ray_hash:,.0f} hashes/seg")

if hashes_encontrados_ray:
    print("💎 Hashes válidos encontrados:")
    for nonce, hash_val in hashes_encontrados_ray[:3]:
        print(f"   Nonce {nonce}: {hash_val}")

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

# ========== RESUMEN FINAL EXTREMO ==========
print("\n" + "="*70)
print("🔥 RESUMEN FINAL - COMPUTACIÓN EXTREMA")
print("="*70)

ejemplos_extremos = [
    ("Factorización Semi-Primos", tiempo_seq_fact, tiempo_ray_fact, "números de 18 dígitos"),
    ("Hash Mining Crypto", tiempo_seq_hash, tiempo_ray_hash, f"{total_nonces:,} hashes"),
    ("Análisis Científico", tiempo_seq_data, tiempo_ray_data, f"{tamano_dataset:,} datos")
]

print(f"{'Ejemplo':<25} {'Secuencial':<12} {'Ray':<12} {'Speedup':<12} {'Descripción'}")
print("-" * 85)

total_tiempo_seq = 0
total_tiempo_ray = 0
speedups_extremos = []

for nombre, t_seq, t_ray, desc in ejemplos_extremos:
    speedup = t_seq / t_ray
    speedups_extremos.append(speedup)
    total_tiempo_seq += t_seq
    total_tiempo_ray += t_ray
    print(f"{nombre:<25} {t_seq:<12.1f} {t_ray:<12.1f} {speedup:<12.2f}x {desc}")

speedup_total = total_tiempo_seq / total_tiempo_ray
print("-" * 85)
print(f"{'TOTAL EXTREMO':<25} {total_tiempo_seq:<12.1f} {total_tiempo_ray:<12.1f} {speedup_total:<12.2f}x")
print(f"Tiempo total ahorrado: {(total_tiempo_seq - total_tiempo_ray)/60:.1f} minutos")

# Estadísticas del sistema bajo carga extrema
print(f"\n💻 RENDIMIENTO DEL SISTEMA BAJO CARGA EXTREMA:")
print(f"CPUs utilizadas: {num_cpus}")
print(f"Speedup promedio: {np.mean(speedups_extremos):.2f}x")
print(f"Eficiencia promedio: {np.mean(speedups_extremos)/num_cpus*100:.1f}% del máximo teórico")

if speedup_total > num_cpus * 0.7:
    print(f"🚀 ¡EXCELENTE! Tu sistema está aprovechando Ray al máximo")
    print(f"💪 Ray está usando eficientemente {num_cpus} cores")
elif speedup_total > num_cpus * 0.4:
    print(f"👍 Buen rendimiento con Ray")
    print(f"🔧 Hay espacio for mejora en la paralelización")
else:
    print(f"⚠️  Rendimiento limitado. Posibles causas:")
    print("   - Sistema con pocos cores o sobrecargado")
    print("   - Limitaciones de memoria")
    print("   - Overhead de Ray significativo")

print(f"\n⏱️  TIEMPO TOTAL DE EJECUCIÓN: {(total_tiempo_ray)/60:.1f} minutos")
print(f"🔥 Has completado la prueba de computación EXTREMA con Ray!")

# Limpiar
ray.shutdown()
print("\n🧹 Ray cerrado exitosamente")
print("✅ Demo EXTREMO completado - ¡Tu CPU necesita un descanso! 🌡️")

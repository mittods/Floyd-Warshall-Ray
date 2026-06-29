import ray
import time
import numpy as np


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
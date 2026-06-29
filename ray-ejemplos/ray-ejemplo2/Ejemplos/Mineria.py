import ray
import time
import numpy as np
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

# CONFIGURACION DE PRUEBA

# Ejemplo 2: DIFICULTAD DE HASH MINING
dificultad = 5  # Cantidad de 0 iniciales en el hash | Default: 5 (mayor a 5 es mas complicado)

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
dificultad = 10  # Hash debe empezar con 5 ceros
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
    ready, futures = ray.wait(futures, num_returns=1, timeout=10)
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
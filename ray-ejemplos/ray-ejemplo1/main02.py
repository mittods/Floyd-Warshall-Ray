import time 
import ray 
import os 

# Inicializar Ray
#ray.init(ignore_reinit_error= True ) 
ray.init(_node_ip_address='172.16.115.137')

# Función para comprobar si un número es primo 
def  is_prime ( n ): 
    if n <= 1 : 
        return  False 
    for i in  range ( 2 , int (n ** 0.5 ) + 1 ): 
        if n % i == 0 : 
            return  False 
    return  True 

# Función paralelizada para encontrar primos en un rango usando Ray 
@ray.remote 
def  find_primes_in_range_parallel ( start, end ): 
    primes = [] 
    for number in  range (start, end): 
        if is_prime(number): 
            primes.append(number) 
    return primes 

if __name__ == "__main__" : 
    start_time = time.time() 

    # Definir rango
    start = 1
    end = 20000000  # Encontrar primos entre 1 y 20 millones
    num_splits = os.cpu_count() - 1   # Número de divisiones (tareas paralelas) 

    # Divide el rango en fragmentos más pequeños para procesamiento paralelo
    range_splits = [(i, i + (end - start) // num_splits) for i in  range (start, end, (end - start) // num_splits)] 
    print(range_splits)

    # Usa Ray para encontrar primos en paralelo
    results = ray.get([find_primes_in_range_parallel.remote(split_start, split_end) for split_start, split_end in range_splits]) 

    # Combina los resultados
    primes = [prime for sublist in results for prime in sublist] 

    end_time = time.time() 
    elapse_time = end_time - start_time
    print ( f'Tiempo tomado: {elapse_time:.2f} segundos' ) 
    print ( f"Número de primos encontrados: { len(primes)} " )

    # Apagar Ray
    ray.shutdown()
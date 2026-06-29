import time 

# Función para comprobar si un número es primo 
def  is_prime ( n ): 
    if n <= 1 : 
        return  False 
    for i in  range ( 2 , int (n ** 0.5 ) + 1 ): 
        if n % i == 0 : 
            return  False 
    return  True 

# Función para encontrar primos en un rango 
def  find_primes_in_range ( start, end ): 
    primes = [] 
    for number in  range (start, end): 
        if is_prime(number): 
            primes.append(number) 
    return primes 

if __name__ == "__main__" : 
    start_time = time.time() 

    # Definir rango (ajustar el rango para el rendimiento de su computadora portátil)
    start = 1
    end = 20000000   # Encontrar primos entre 1 y 20 millones 

    # Encontrar primos
    primes = find_primes_in_range(start, end) 

    end_time = time.time() 
    elapse_time = end_time - start_time
    print ( f'Tiempo tomado: {elapse_time:.2f} segundos' ) 
    print ( f"Número de primos encontrados: { len(primes)} " )
  
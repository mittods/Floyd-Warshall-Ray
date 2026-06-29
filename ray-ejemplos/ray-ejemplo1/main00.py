import ray 

ray.init()   # Inicializar Ray 

@ray.remote 
def  square ( x ): 
    return x * x 

# Lanzar tareas en paralelo
futures = [square.remote(i) for i in  range ( 5 )] 
results = ray.get(futures)   # Recuperar resultados 
print (results)   # Salida: [0, 1, 4, 9, 16]


@ray .remote 
class  Contador: 
    def  __init__ ( self ): 
        self .count = 0 
    
    def  increment ( self ): 
        self .count += 1 
        return  self .count 


contador = Contador.remote() 
print(ray.get(contador.increment.remote()))   # Salida: 1
print(ray.get(contador.increment.remote()))   # Salida: 2
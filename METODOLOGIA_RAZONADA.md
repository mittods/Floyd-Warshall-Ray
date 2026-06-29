# Metodología Razonada: Paralelización de Floyd-Warshall con Ray

**Documento de razonamiento metodológico — No es parte del artículo**

Escrito desde la perspectiva del investigador. Describe el proceso completo
de toma de decisiones para diseñar este experimento.

---

## 1. Análisis inicial del problema

Al comenzar este trabajo, la pregunta central era aparentemente simple:
*¿puede Ray acelerar Floyd-Warshall?* Sin embargo, esta pregunta oculta varias
sub-preguntas que determinan completamente el diseño experimental:

1. ¿Qué partes del algoritmo son paralelizables y cuáles no?
2. ¿Qué estrategia de paralelización con Ray es técnicamente adecuada?
3. ¿Qué métricas medir y con qué herramientas?
4. ¿Qué tamaños de problema y configuraciones permiten una caracterización completa?
5. ¿Cómo garantizar que los resultados sean reproducibles por cualquier investigador?

La primera tarea fue estudiar el algoritmo desde la perspectiva de las
dependencias de datos, no solo de la complejidad computacional.

### 1.1 Análisis de dependencias del algoritmo

Floyd-Warshall opera sobre una matriz de distancias `dist[n][n]`. El cuerpo
del algoritmo es:

```
para k = 0 .. n-1:
    para i = 0 .. n-1:
        para j = 0 .. n-1:
            dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])
```

La pregunta crítica es: **¿qué dependencias existen entre las iteraciones?**

**Entre iteraciones de j (bucle más interno):** Las actualizaciones de
`dist[i][j]` para distintos `j` son completamente independientes dado un `k` e `i`
fijos. `dist[i][k]` y `dist[k][j]` son de **solo lectura** para `j` variable.
→ *Paralelizable trivialmente.*

**Entre iteraciones de i (bucle medio):** Las actualizaciones de distintas
filas `i` son completamente independientes para un `k` fijo. `dist[i][k]`
varía por fila pero `dist[k][j]` es compartido como solo lectura.
→ *Paralelizable: la unidad natural de trabajo es una fila completa.*

**Entre iteraciones de k (bucle externo):** Aquí está la restricción fundamental.
La iteración `k+1` requiere `dist[i][k+1]` que puede haber sido **modificado**
en la iteración `k` (cuando se actualizó `dist[i][k+1]` con `k` como intermedio).
→ **No paralelizable: el bucle externo sobre k debe ser secuencial.**

Esta restricción es inherente al algoritmo y no puede eliminarse sin cambiar
la semántica del algoritmo (algo que las variantes como Dijkstra o Bellman-Ford
abordan con diferentes trade-offs).

**Implicación directa para el speedup teórico:** La fracción no paralelizable
es exactamente la sincronización de las `n` iteraciones de `k`. En términos
de la Ley de Amdahl: si el tiempo de sincronización por iteración `k` es `s`
y el tiempo de cómputo por iteración es `t`, entonces:
- Tiempo secuencial: `n * (s + n²)`  [donde n² representa las n² actualizaciones]
- Tiempo paralelo (p workers): `n * (s + n²/p)`
- Speedup máximo: `(s + n²) / (s + n²/p)`

Para `s << n²` (matrices grandes), el speedup se acerca a `p`. Para `s >> n²`
(matrices pequeñas), el speedup tiende a 1. **Esto predice exactamente el
comportamiento crossover que buscamos medir experimentalmente.**

---

## 2. Alternativas consideradas para paralelizar Floyd-Warshall con Ray

Identifiqué cuatro estrategias posibles:

### Alternativa A: Tareas sin estado (task-based, sin actores)

Para cada `k`, crear `p` tareas remotas que reciban su chunk de filas,
procesen y retornen las filas actualizadas.

**Ventajas:** Conceptualmente simple, fácil de implementar.  
**Desventajas críticas:**
- Cada iteración `k` transfiere O(n²/p) datos al object store (envío del chunk)
  y otros O(n²/p) datos en el retorno. Total: O(n²) datos por iteración k.
- Para n=1024 y 32 workers: 1024 iteraciones × 8 MB/iteración ≈ 8 GB transferidos.
- Tiempo de serialización dominaría completamente el tiempo de cómputo.

**Conclusión:** Descartada por excesiva transferencia de datos.

### Alternativa B: Actores con estado (actor-based, ELEGIDA)

Crear `p` actores, cada uno manteniendo su bloque de filas como estado local.
Solo la fila `k` se transfiere (O(n) por iteración).

**Ventajas:**
- Transferencia mínima: solo la fila k (n floats = 8n bytes) por iteración.
- Para n=1024: 1024 × 8 KB ≈ 8 MB total (vs. 8 GB de la alternativa A).
- La columna k la extrae cada actor de su propio bloque sin transferencia.
- Modelo conceptualmente claro: cada actor "es dueño" de sus filas.

**Desventajas:**
- Mayor complejidad de implementación.
- Overhead de inicialización de actores (costo fijo, amortizable).
- Hay `n` puntos de sincronización (uno por iteración k), lo que puede
  ser costoso si la latencia del scheduler de Ray es significativa.

**Conclusión:** Esta es la estrategia óptima para la arquitectura de Ray.

### Alternativa C: Variante en bloques (blocked Floyd-Warshall)

La variante de Venkataraman et al. (2003) restructura el algoritmo en
operaciones de bloques de submatrices para mejorar la localidad de caché.
En una arquitectura distribuida, cada bloque podría asignarse a un actor.

**Ventajas:** Mejor localidad de caché, posibilidad de mayor paralelismo.  
**Desventajas:** Complejidad de implementación muy superior; la dependencia
entre bloques dentro de cada iteración `k` complica la paralelización correcta.

**Conclusión:** Reservada como trabajo futuro. Implementar para el artículo
inicial aumentaría el scope innecesariamente.

### Alternativa D: Paralelización de filas dentro de cada actor

Un actor podría usar múltiples hilos internamente (NumPy/MKL ya hace esto
automáticamente para operaciones vectorizadas grandes).

**Ventajas:** Potencialmente más eficiente para bloques grandes.  
**Desventajas:** Interfiere con el control de recursos de Ray; BLAS/MKL
puede crear hilos propios independientemente del control de Ray.

**Conclusión:** No se controla explícitamente, pero NumPy aprovechará
las instrucciones SIMD (AVX-512 en el Threadripper) automáticamente.

---

## 3. Justificación de la estrategia Ray elegida

Elegí la **Alternativa B (actores con estado)** por las siguientes razones:

1. **Mínima comunicación:** Solo `n` vectores de tamaño `n` se transfieren
   durante toda la ejecución (la fila k en cada iteración). El resto de los
   datos permanece en memoria local de cada actor.

2. **Modelo intuitivo:** El concepto de "actor dueño de sus filas" es fácil
   de explicar y verificar. La correctitud es inmediata de razonar.

3. **Object store de Ray:** Al hacer `ray.put(row_k)`, Ray almacena la fila
   en memoria compartida accesible por todos los actores del mismo nodo sin
   copia adicional. Esto es una ventaja significativa de Ray vs. multiprocessing.

4. **Alineación con el modelo de actores de Ray:** La clase `FilasActor` con
   `@ray.remote` es exactamente el caso de uso para el que Ray está optimizado.
   La alternativa de tareas stateless es menos eficiente para este patrón.

---

## 4. Decisiones de diseño del software

### 4.1 Separación secuencial/paralelo en módulos distintos

Podría haberse implementado todo en un único archivo, pero la separación en
`src/secuencial/` y `src/ray_parallel/` permite:
- Comparar implementaciones sin riesgo de contaminación entre ellas.
- Importar independientemente en tests de correctitud.
- Comunicar claramente cuál es la "línea base" y cuál la "mejora propuesta".

### 4.2 Vectorización NumPy en la versión secuencial

Decidí optimizar la implementación secuencial con broadcasting NumPy en lugar
de usar tres bucles Python explícitos. **Esta decisión es deliberada y requiere
justificación:**

Si usara bucles Python puros, la versión secuencial sería artificialmente lenta
(por el GIL y el overhead de la interpretación Python), lo que inflaría
artificialmente el speedup de Ray. El speedup reportado no reflejaría el
beneficio real del paralelismo, sino simplemente la comparación con una
implementación ineficiente.

Al vectorizar la línea base, medimos el **speedup real**: cuánto agrega Ray
sobre una implementación Python de producción razonable.

### 4.3 Número de actores vs. número de CPUs

Los scripts y la configuración permiten especificar el número de actores
independientemente del número de CPUs del sistema. Esto es intencional:
- Permite estudiar el efecto de sobresubscripción (más actores que CPUs).
- Permite medir el overhead de coordinación con pocos actores.
- El hardware objetivo (64 hilos) nos permite probar hasta 32 actores sin
  sobresubscripción de núcleos físicos.

### 4.4 Verificación de correctitud integrada

El benchmark verifica automáticamente que la versión Ray produce el mismo
resultado que la secuencial (con `np.allclose`). Esto no es overhead experimental:
es parte del protocolo científico que garantiza que medimos el algoritmo correcto.

---

## 5. Justificación de Docker

La reproducibilidad científica es un requisito no negociable. Sin un entorno
controlado, los experimentos podrían no ser reproducibles por:

1. **Diferencias de versiones:** Ray 2.44 vs. Ray 2.0 tienen diferencias
   significativas en el overhead del scheduler.
2. **Diferencias del sistema:** Ubuntu vs. Arch Linux vs. Debian pueden tener
   distintos backends de BLAS (OpenBLAS vs. MKL vs. ATLAS), afectando
   el rendimiento de NumPy.
3. **Bibliotecas del sistema:** Las versiones de libc, libpthread afectan
   el rendimiento de operaciones atómicas usadas por Ray.

Docker garantiza que cualquier investigador con el mismo hardware obtendrá
exactamente el mismo entorno de software. La publicación en GHCR mediante
GitHub Actions elimina la necesidad de construir la imagen localmente.

**Por qué GHCR y no DockerHub:** GHCR es nativo de GitHub, lo que permite
integración directa con los permisos del repositorio y no requiere secrets
adicionales en Actions. La imagen se vincula automáticamente al repositorio.

---

## 6. Justificación de las métricas y herramientas

### 6.1 Tiempo con `time.perf_counter()`

- Resolución: subnanosegundo en Linux (usa `clock_gettime(CLOCK_MONOTONIC)` internamente).
- No afectado por ajustes de reloj del sistema (NTP, leap seconds).
- Alternativa descartada: `time.time()` tiene menor resolución y puede saltar.

### 6.2 CPU con `psutil`

- Biblioteca multiplataforma con mínimo overhead.
- `cpu_percent(percpu=True)` provee uso por núcleo, importante para verificar
  que Ray está distribuyendo el trabajo efectivamente.
- Alternativa considerada: `/proc/stat` directamente. Más bajo nivel pero
  sin ventaja práctica dado que psutil ya lee de ahí.

### 6.3 GPU con `pynvml`

Aunque Floyd-Warshall corre en CPU, registrar el estado de la GPU sirve para:
1. Documentar que la GPU no participa en la ejecución (validez del experimento).
2. Detectar si algún otro proceso usa la GPU durante el benchmark (amenaza a la validez).
3. Preparar la infraestructura para extensiones futuras (ej. cuPy).

`pynvml` vs. `subprocess + nvidia-smi`: pynvml es más eficiente (sin fork/exec
por cada muestra), más robusto (manejo de errores tipado) y provee la misma info.

### 6.4 Energía con RAPL (`/sys/class/powercap/`)

El hardware AMD Ryzen Threadripper PRO 5975WX soporta la interfaz RAPL a
través del driver `amd_energy` (disponible en kernel ≥ 5.8). Los contadores
se leen diferencialmente (inicio y fin de la ejecución) para obtener la
energía consumida durante el algoritmo.

**Limitación conocida:** Requiere permisos de lectura en `/sys/class/powercap/`
(normalmente requiere root o grupo `power`). El código implementa un fallback
graceful si no está disponible.

### 6.5 Número de repeticiones y test de Grubbs

**¿Por qué 10 repeticiones?**

Con 10 muestras y 9 grados de libertad, el intervalo de confianza al 95%
tiene un factor `t_{0.025, 9} = 2.262`. La amplitud del intervalo es
`2 × 2.262 × σ/√10 ≈ 1.43 × σ`. Con semillas distintas para cada repetición,
capturamos la variabilidad debida a diferentes estructuras de grafo.

Con 30 o más muestras usaríamos el z de la distribución normal; con 10 la
distribución t es más apropiada (colas más pesadas = intervalos más conservadores).

**¿Por qué Grubbs y no IQR?**

El test de Grubbs está diseñado específicamente para detectar outliers en
muestras pequeñas (n=10) asumiendo distribución normal. El método IQR es
más adecuado para muestras grandes. Con 10 repeticiones y tiempos que
generalmente siguen distribuciones aproximadamente normales, Grubbs es la
elección estadísticamente apropiada.

---

## 7. Justificación de los tamaños de entrada

Los tamaños elegidos son: {64, 128, 256, 512, 1024, 2048, 4096}.

**¿Por qué potencias de 2?**

1. Permiten partición exacta entre workers (sin filas "sobrantes") para
   cualquier potencia de 2 de workers (1, 2, 4, 8, 16, 32).
2. El comportamiento de la caché L1/L2/L3 cambia en puntos específicos
   relacionados con potencias de 2.

**¿Por qué comenzar en n=64?**

Para `n=64`, la matriz ocupa solo 32 KB (float64), que cabe completamente
en la caché L1 (normalmente 32-64 KB por núcleo). El overhead de Ray
dominará completamente. Esto es exactamente el punto de operación donde
queremos demostrar que Ray no es beneficioso.

**¿Por qué terminar en n=4096?**

Para `n=4096`, la matriz ocupa 128 MB (float64), que supera la caché L3
del Threadripper PRO 5975WX (típicamente 128-256 MB). La versión secuencial
tendrá muchos cache misses. Queremos verificar si Ray mejora esto o si el
overhead supera la ganancia de distribución de memoria.

Memoria total disponible: 128 GB. La matriz de n=4096 en float64 es 128 MB,
por lo que incluso con 32 actores (32 × 128 MB ≈ 4 GB para las particiones)
hay amplio margen.

---

## 8. Amenazas a la validez

### 8.1 Amenazas a la validez interna

**Ruido del sistema operativo:** El scheduler del SO puede interrumpir el
proceso durante mediciones. Mitigación: 10 repeticiones + test de Grubbs.

**Calentamiento de caché:** Las primeras repeticiones pueden ser más lentas
por caché fría. Mitigación: cada repetición usa una semilla diferente (grafo
distinto), por lo que no hay calentamiento acumulativo intencionado.

**Variación de frecuencia de CPU (turbo boost):** El AMD Threadripper PRO 5975WX
usa boost adaptativo. Mitigación: registrar la frecuencia durante la ejecución
con `psutil.cpu_freq()` y documentar las variaciones.

**Estado inicial de Ray:** Si Ray tiene objetos residuales en el object store,
puede afectar el rendimiento. Mitigación: Ray se inicializa una sola vez al
inicio del benchmark y los actores se crean frescos para cada experimento.

### 8.2 Amenazas a la validez externa

**Hardware específico:** El Threadripper PRO 5975WX tiene características
únicas (128 MB RAM HBM2 en GPU, arquitectura Zen 3 con NUMA). Los resultados
pueden no generalizarse directamente a otras CPUs.

**Sistema operativo:** El entorno Docker sobre Linux con kernel específico
puede diferir de otros sistemas. La imagen Docker mitiga esto para replicación.

**Versión de Ray:** Ray 2.44 tiene un scheduler diferente a versiones anteriores.
Los resultados son específicos a esta versión. Mitigación: la imagen Docker
fija exactamente la versión.

### 8.3 Amenazas a la validez de constructo

**Definición de speedup:** Usamos `T_seq / T_ray` donde `T_seq` es la
versión NumPy vectorizada. Esto mide el beneficio real de Ray, no el
speedup teórico del algoritmo.

**Overhead de Ray incluido:** `T_ray` incluye la inicialización de actores y
la recolección de resultados. Esto es correcto para medir el beneficio neto,
pero puede compararse con el costo de cómputo puro para análisis de overhead.

---

## 9. Limitaciones

1. **Sin comparación con OpenMP/MPI:** Sería valioso comparar Ray con
   alternativas de bajo nivel (OpenMP, MPI, numba.jit con parallel=True).
   No se incluye para mantener el scope del artículo enfocado en Ray.

2. **Single-node únicamente:** Ray está diseñado para clusters multi-nodo,
   pero este experimento evalúa solo el caso single-node. La escalabilidad
   multi-nodo de Floyd-Warshall con Ray es un estudio separado.

3. **Sin GPU:** Floyd-Warshall en GPU (cuPy) podría ofrecer speedups de
   10-40×. La comparación con la versión Ray-CPU es una limitación del scope.

4. **Densidad de grafo fija:** Se evalúa con densidad 0.7 (70% de aristas).
   Grafos dispersos vs. densos pueden tener comportamientos distintos,
   especialmente respecto a la distribución de INF en la matriz.

---

## 10. Posibles mejoras futuras

### Mejoras algorítmicas

- **Blocked Floyd-Warshall con Ray:** Reestructurar en operaciones de
  bloques para mejor localidad de caché y paralelismo más granular.
- **Versión cuPy:** Implementar usando cuPy para la NVIDIA TITAN V y
  comparar con la versión CPU Ray.
- **Algoritmo de Dijkstra paralelo con Ray:** Para grafos sin pesos negativos,
  comparar el rendimiento con una variante basada en Dijkstra.

### Mejoras de infraestructura

- **Ray distribuido real:** Evaluar en un cluster de varios nodos con
  Ray en modo head+worker.
- **Profiling con py-spy:** Obtener flame graphs del tiempo por función
  para identificar cuellos de botella no evidentes.
- **Benchmarks de memoria con Valgrind/massif:** Análisis detallado del
  perfil de memoria para entender el overhead del object store de Ray.

### Mejoras metodológicas

- **Comparación con Numba:** `@numba.jit(parallel=True)` podría ofrecer
  paralelismo con menor overhead que Ray para el caso single-node.
- **Análisis de sensibilidad de densidad:** Repetir experimentos con
  densidades 0.3, 0.5, 0.7, 0.9 para caracterizar el impacto.
- **Warmup estadístico:** Agregar una repetición de warmup descartada para
  eliminar efectos de primera ejecución.

---

*Documento generado el 2026-06-29.*  
*Autores: [COMPLETAR]*

"""
Monitor de recursos del sistema: CPU, RAM, GPU y energía.

Utiliza un hilo de fondo para muestrear métricas a intervalos regulares
durante la ejecución del algoritmo. Las muestras se agregan al finalizar.

Herramientas utilizadas y justificación:
    - psutil: Biblioteca multiplataforma para CPU, RAM y procesos.
              Ampliamente mantenida, mínimo overhead, sin necesidad de root.
    - pynvml: Binding Python oficial de NVML (NVIDIA Management Library).
              Acceso a GPU utilization, memoria, potencia y temperatura
              de la NVIDIA TITAN V sin invocar subprocesos.
    - /sys/class/powercap/: Interfaz del kernel para RAPL (AMD/Intel).
              En AMD Ryzen Threadripper PRO 5975WX, disponible vía
              amd_energy o k10temp. Se intenta, con fallback a 0.0 si
              no está accesible (requiere privilegios o driver específico).
"""
import os
import time
import logging
import threading
from typing import Optional

import psutil
import numpy as np

logger = logging.getLogger(__name__)

# Intentar importar pynvml; si no está disponible, GPU desactivada
try:
    import pynvml
    pynvml.nvmlInit()
    _NVML_DISPONIBLE = True
except Exception:
    _NVML_DISPONIBLE = False
    logger.warning("pynvml no disponible: métricas GPU desactivadas.")


def _leer_energia_rapl() -> Optional[float]:
    """
    Lee el consumo acumulado de energía CPU desde /sys/class/powercap/.

    Retorna energía en Joules o None si no está disponible.
    Funciona con Intel RAPL y AMD Energy (kernel ≥ 5.8).
    """
    rutas = [
        "/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj",
        "/sys/class/powercap/amd_energy/amd_energy:0/energy_uj",
    ]
    for ruta in rutas:
        try:
            with open(ruta) as f:
                return int(f.read().strip()) / 1e6  # μJ → J
        except (FileNotFoundError, PermissionError, ValueError):
            continue
    return None


class MonitorSistema:
    """
    Monitor de recursos que muestrea en segundo plano durante la ejecución.

    Uso:
        monitor = MonitorSistema(intervalo_s=0.5)
        with monitor:
            resultado = ejecutar_algoritmo(...)
        metricas_sistema = monitor.obtener_metricas()
    """

    def __init__(self, intervalo_s: float = 0.5, pid: Optional[int] = None):
        """
        Args:
            intervalo_s: Intervalo de muestreo en segundos.
            pid: PID del proceso a monitorear. None → proceso actual.
        """
        self.intervalo_s = intervalo_s
        self.pid = pid or os.getpid()
        self._proceso = psutil.Process(self.pid)

        self._hilo: Optional[threading.Thread] = None
        self._detener = threading.Event()

        # Buffers de muestras
        self._muestras_cpu: list[float] = []
        self._muestras_cpu_por_nucleo: list[list[float]] = []
        self._muestras_ram_total_mb: list[float] = []
        self._muestras_ram_proceso_mb: list[float] = []
        self._muestras_gpu_uso: list[float] = []
        self._muestras_gpu_mem_mb: list[float] = []
        self._muestras_gpu_temp: list[float] = []
        self._muestras_gpu_potencia_w: list[float] = []

        # Energía (medición diferencial inicio/fin)
        self._energia_cpu_inicio_j: Optional[float] = None
        self._energia_cpu_fin_j: Optional[float] = None
        self._t_inicio: float = 0.0
        self._t_fin: float = 0.0

        # Handle GPU si disponible
        self._gpu_handle = None
        if _NVML_DISPONIBLE:
            try:
                self._gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            except pynvml.NVMLError:
                logger.warning("No se pudo obtener handle de GPU.")

    def _muestrear(self) -> None:
        """Bucle de muestreo que corre en el hilo de fondo."""
        # Primera llamada para inicializar el porcentaje diferencial
        psutil.cpu_percent(percpu=True)
        time.sleep(self.intervalo_s)

        while not self._detener.is_set():
            try:
                # CPU
                uso_global = psutil.cpu_percent(interval=None)
                uso_por_nucleo = psutil.cpu_percent(percpu=True, interval=None)
                self._muestras_cpu.append(uso_global)
                self._muestras_cpu_por_nucleo.append(uso_por_nucleo)

                # RAM total del sistema
                mem = psutil.virtual_memory()
                self._muestras_ram_total_mb.append(
                    (mem.total - mem.available) / 1024**2
                )

                # RAM del proceso
                try:
                    info_proc = self._proceso.memory_info()
                    self._muestras_ram_proceso_mb.append(
                        info_proc.rss / 1024**2
                    )
                except psutil.NoSuchProcess:
                    pass

                # GPU
                if self._gpu_handle is not None:
                    try:
                        util = pynvml.nvmlDeviceGetUtilizationRates(self._gpu_handle)
                        mem_info = pynvml.nvmlDeviceGetMemoryInfo(self._gpu_handle)
                        temp = pynvml.nvmlDeviceGetTemperature(
                            self._gpu_handle, pynvml.NVML_TEMPERATURE_GPU
                        )
                        potencia_mw = pynvml.nvmlDeviceGetPowerUsage(self._gpu_handle)
                        self._muestras_gpu_uso.append(float(util.gpu))
                        self._muestras_gpu_mem_mb.append(mem_info.used / 1024**2)
                        self._muestras_gpu_temp.append(float(temp))
                        self._muestras_gpu_potencia_w.append(potencia_mw / 1000.0)
                    except pynvml.NVMLError:
                        pass

            except Exception as e:
                logger.debug("Error en muestreo: %s", e)

            time.sleep(self.intervalo_s)

    def iniciar(self) -> None:
        """Inicia el monitor en segundo plano."""
        self._detener.clear()
        self._energia_cpu_inicio_j = _leer_energia_rapl()
        self._t_inicio = time.perf_counter()

        self._hilo = threading.Thread(target=self._muestrear, daemon=True)
        self._hilo.start()

    def detener(self) -> None:
        """Detiene el muestreo y recopila la lectura final de energía."""
        self._detener.set()
        self._t_fin = time.perf_counter()
        self._energia_cpu_fin_j = _leer_energia_rapl()
        if self._hilo:
            self._hilo.join(timeout=2.0)

    def __enter__(self) -> "MonitorSistema":
        self.iniciar()
        return self

    def __exit__(self, *_) -> None:
        self.detener()

    def obtener_metricas(self) -> dict:
        """
        Agrega todas las muestras en un diccionario de métricas.

        Retorna:
            dict con estadísticas promedio/máximo de todos los recursos.
        """
        duracion_s = self._t_fin - self._t_inicio

        # CPU
        cpu_arr = np.array(self._muestras_cpu) if self._muestras_cpu else np.array([0.0])
        nucleo_prom: list[float] = []
        if self._muestras_cpu_por_nucleo:
            arr_nucleo = np.array(self._muestras_cpu_por_nucleo)
            nucleo_prom = arr_nucleo.mean(axis=0).tolist()

        # RAM
        ram_arr = np.array(self._muestras_ram_total_mb) if self._muestras_ram_total_mb else np.array([0.0])
        ram_proc_arr = np.array(self._muestras_ram_proceso_mb) if self._muestras_ram_proceso_mb else np.array([0.0])
        ram_total_gb = psutil.virtual_memory().total / 1024**3

        # Frecuencia CPU
        freq_info = psutil.cpu_freq()
        freq_mhz = freq_info.current if freq_info else 0.0

        # Hilos activos
        try:
            hilos = self._proceso.num_threads()
        except psutil.NoSuchProcess:
            hilos = 0

        # GPU
        gpu_disponible = bool(self._muestras_gpu_uso)
        gpu_uso_prom = float(np.mean(self._muestras_gpu_uso)) if self._muestras_gpu_uso else 0.0
        gpu_mem_prom = float(np.mean(self._muestras_gpu_mem_mb)) if self._muestras_gpu_mem_mb else 0.0
        gpu_temp_prom = float(np.mean(self._muestras_gpu_temp)) if self._muestras_gpu_temp else 0.0
        gpu_pot_prom_w = float(np.mean(self._muestras_gpu_potencia_w)) if self._muestras_gpu_potencia_w else 0.0
        gpu_energia_j = gpu_pot_prom_w * duracion_s

        # Energía CPU
        cpu_energia_j = 0.0
        if (
            self._energia_cpu_inicio_j is not None
            and self._energia_cpu_fin_j is not None
            and self._energia_cpu_fin_j >= self._energia_cpu_inicio_j
        ):
            cpu_energia_j = self._energia_cpu_fin_j - self._energia_cpu_inicio_j

        cpu_potencia_w = cpu_energia_j / duracion_s if duracion_s > 0 else 0.0
        energia_total_j = cpu_energia_j + gpu_energia_j

        return {
            # CPU
            "cpu_uso_promedio_pct": float(cpu_arr.mean()),
            "cpu_uso_maximo_pct": float(cpu_arr.max()),
            "cpu_uso_por_nucleo": nucleo_prom,
            "cpu_hilos_activos": hilos,
            "cpu_frecuencia_mhz": float(freq_mhz),
            # RAM
            "ram_pico_mb": float(ram_arr.max()),
            "ram_promedio_mb": float(ram_arr.mean()),
            "ram_proceso_pico_mb": float(ram_proc_arr.max()),
            "ram_total_gb": float(ram_total_gb),
            # GPU
            "gpu_disponible": gpu_disponible,
            "gpu_uso_pct": gpu_uso_prom,
            "gpu_memoria_mb": gpu_mem_prom,
            "gpu_temperatura_c": gpu_temp_prom,
            "gpu_potencia_w": gpu_pot_prom_w,
            "gpu_energia_j": float(gpu_energia_j),
            # Energía
            "cpu_potencia_w": float(cpu_potencia_w),
            "cpu_energia_j": float(cpu_energia_j),
            "energia_total_j": float(energia_total_j),
            "potencia_promedio_w": float((cpu_potencia_w + gpu_pot_prom_w)),
        }

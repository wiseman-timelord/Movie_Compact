# Script: `.\scripts\hardware.py`

# Imports...
import pyopencl as cl
import cpuinfo
import sys

# Classes...
class HardwareManager:
    """Centralized hardware capability detection and context management"""
    
    @classmethod
    def detect_capabilities(cls) -> dict:
        """
        Detect hardware capabilities, including AMD OpenCL GPUs and AVX2 CPU support.
        Returns a dictionary with detected features and GPU details.
        """
        caps = {
            'OpenCL': False,
            'AVX2': 'avx2' in cpuinfo.get_cpu_info()['flags'],
            'AOCL': False,
            'x64': sys.maxsize > 2**32,
            'gpu_devices': []
        }
        
        try:
            platforms = cl.get_platforms()
            for platform in platforms:
                if "AMD" in platform.name:
                    devices = platform.get_devices(device_type=cl.device_type.GPU)
                    for device in devices:
                        vram = device.global_mem_size // (1024 ** 3)  # Convert bytes to GB
                        caps['gpu_devices'].append({
                            'name': device.name.strip(),
                            'vram_gb': vram
                        })
                    if devices:
                        caps['OpenCL'] = True
                        caps['AOCL'] = True
        except Exception as e:
            print(f"OpenCL detection failed: {str(e)}")
            
        return caps

    @classmethod
    def create_context(cls) -> dict:
        ctx = {
            'use_opencl': False,
            'use_avx2': True,  # Assume AVX2 is always available per your hardware
            'platform': None,
            'device': None
        }
        
        try:
            platforms = cl.get_platforms()
            for platform in platforms:
                if "AMD" in platform.name:
                    devices = platform.get_devices()
                    for device in devices:
                        if device.type == cl.device_type.GPU and device.global_mem_size >= 8 * 1024**3:
                            ctx.update({
                                'use_opencl': True,
                                'platform': platform,
                                'device': device
                            })
                            break
                    if ctx['use_opencl']:
                        break
        except Exception as e:
            print(f"OpenCL init failed: {e}, falling back to AVX2")
        
        return ctx
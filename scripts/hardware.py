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
        """REPLACES print_hardware_info from launcher.py"""
        caps = {
            'OpenCL': False,
            'AVX2': 'avx2' in cpuinfo.get_cpu_info()['flags'],
            'AOCL': False,
            'x64': sys.maxsize > 2**32
        }
        
        try:
            platforms = cl.get_platforms()
            if platforms:
                caps['OpenCL'] = True
                caps['devices'] = [d.name for d in platforms[0].get_devices()]
        except Exception as e:
            print(f"OpenCL detection failed: {str(e)}")
            
        return caps

    @classmethod
    def create_context(cls) -> dict:
        """REPLACES OpenCL setup from utility.py"""
        caps = cls.detect_capabilities()
        ctx = {
            'use_opencl': False,
            'use_avx2': False,
            'platform': None,
            'device': None
        }
        
        if caps['OpenCL']:
            try:
                platform = cl.get_platforms()[0]
                device = platform.get_devices()[0]
                ctx.update({
                    'use_opencl': True,
                    'platform': platform,
                    'device': device
                })
            except Exception as e:
                print(f"OpenCL init failed: {str(e)}")
        elif caps['AVX2']:
            ctx['use_avx2'] = True
            
        return ctx
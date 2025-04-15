# Script: `.\scripts\hardware.py`

# Imports...
import pyopencl as cl
import cpuinfo
import sys

# Classes...
class HardwareManager:
    """Centralized hardware context management with manual VRAM configuration"""
    
    @classmethod
    def create_context(cls) -> dict:
        """Create processing context with manual VRAM limits"""
        ctx = {
            'vram_limit': ConfigManager.get_vram_limit(),
            'use_opencl': False,
            'use_avx2': False,
            'platform': None,
            'device': None
        }

        # OpenCL detection with manual VRAM limits
        try:
            platforms = cl.get_platforms()
            for platform in platforms:
                if "AMD" in platform.name:
                    devices = platform.get_devices(device_type=cl.device_type.GPU)
                    if devices:
                        # Use first compatible device regardless of reported VRAM
                        ctx.update({
                            'use_opencl': True,
                            'platform': platform,
                            'device': devices[0]
                        })
                        break
        except Exception as e:
            print(f"OpenCL initialization failed: {str(e)}")

        # AVX2 fallback detection
        ctx['use_avx2'] = not ctx['use_opencl'] and cls._has_avx2()
        
        print(f"Hardware context: {ctx}")
        return ctx

    @classmethod
    def _has_avx2(cls):
        """Safe AVX2 detection with fallback"""
        try:
            info = cpuinfo.get_cpu_info()
            return 'avx2' in info.get('flags', [])
        except Exception as e:
            print(f"AVX2 detection failed: {str(e)}")
            return False
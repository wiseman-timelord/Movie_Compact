# Script: `.\scripts\hardware.py`

# Imports...
import pyopencl as cl
import cpuinfo
import sys

# Classes...
class HardwareManager:
    @classmethod
    def create_context(cls) -> dict:
        """Create processing context respecting user configuration"""
        ctx = {
            'vram_limit': ConfigManager.get_vram_limit(),
            'use_opencl': ConfigManager.get('hardware', 'opencl_enabled', False),
            'use_avx2': ConfigManager.get('hardware', 'avx2_fallback', False),
            'platform': None,
            'device': None,
            'context': None
        }

        # OpenCL detection only if enabled in config
        if ctx['use_opencl']:
            try:
                platforms = cl.get_platforms()
                platform_preference = ConfigManager.get('processing', 'hardware_acceleration.opencl_platform_preference', 
                                                      ['NVIDIA', 'AMD', 'Intel'])
                for pref in platform_preference:
                    for platform in platforms:
                        if pref in platform.name:
                            devices = platform.get_devices(cl.device_type.GPU)
                            if devices:
                                ctx.update({
                                    'platform': platform,
                                    'device': devices[0],
                                    'context': cl.Context(devices=[devices[0]])
                                })
                                break
                    if ctx['device']:
                        break
            except Exception as e:
                print(f"OpenCL init failed: {e}")
                ctx['use_opencl'] = False

        # AVX2 fallback handling
        if not ctx['use_opencl'] and ctx['use_avx2']:
            ctx['use_avx2'] = cls._has_avx2()
        else:
            ctx['use_avx2'] = False

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
# Script: `/\scripts/temporary.py`

# Imports...
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple
import os
import json

# Globals...
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Maps...
PROCESSING_CONFIG = {
    'preview_height': 360,
    'target_height': 720,
    'frame_sample_rate': 30,
    'supported_formats': ['.mp4', '.avi', '.mkv'],
    'output_codec': 'libx264',
    'audio_codec': 'aac',
    'hardware_acceleration': {
        'use_gpu': True,
        'use_opencl': True,
        'use_avx2': True,
        'opencl_platform_preference': ['NVIDIA', 'AMD', 'Intel']
    },
    'performance': {
        'gpu_batch_size': 32,
        'cpu_threads': 4,
        'frame_buffer_size': 30,
        'max_parallel_processes': 2
    },
    'scene_detection': {
        'static_threshold': 0.98,
        'menu_threshold': 0.85,
        'motion_threshold': 0.3,
        'texture_threshold': 0.4,
        'min_scene_duration': 2.0,
        'max_scene_duration': 300.0,
        'scene_similarity_threshold': 0.85
    }
}

AUDIO_CONFIG = {
    'sample_rate': 44100,
    'window_size': 2048,
    'hop_length': 512,
    'preserve_pitch': True,
    'enhance_audio': True,
    'feature_settings': {
        'min_duration': 0.1,
        'max_duration': 30.0,
        'energy_threshold': 0.7,
        'frequency_threshold': 0.6
    }
}

MEMORY_CONFIG = {
    'max_memory_usage': 0.8,
    'warning_threshold': 0.7,
    'critical_threshold': 0.9,
    'cleanup_interval': 60,
    'frame_buffer_size': 30,
    'chunk_size_mb': 64
}

PATHS_CONFIG = {
    'input': 'input',
    'output': 'output',
    'work': 'data/temp',
    'data': 'data'
}

SPEED_CONFIG = {
    'max_speed_factor': 4.0,
    'min_speed_factor': 1.0,
    'transition_frames': 30,
    'smooth_window': 15,
    'action_speed': 1.0,
    'menu_speed': 2.0,
    'static_speed': 4.0
}

# Classes...
class ConfigManager:
    """REPLACES all direct config dictionary access"""
    _configs = {
        'processing': PROCESSING_CONFIG,
        'audio': AUDIO_CONFIG,
        'memory': MEMORY_CONFIG,
        'paths': PATHS_CONFIG,
        'speed': SPEED_CONFIG
    }

    @classmethod
    def get(cls, category: str, key: str, default=None):
        """Get config value using dot-notation keys"""
        keys = key.split('.')
        value = cls._configs.get(category.lower(), {})
        
        for k in keys:
            value = value.get(k, {})
            if not isinstance(value, dict):
                break
                
        return value if value != {} else default

    @classmethod
    def update(cls, category: str, updates: dict):
        """Deep update for nested config structures"""
        category = category.lower()
        if category not in cls._configs:
            raise ValueError(f"Invalid config category: {category}")
            
        def _update(d, u):
            for k, v in u.items():
                if isinstance(v, dict):
                    d[k] = _update(d.get(k, {}), v)
                else:
                    d[k] = v
            return d
            
        cls._configs[category] = _update(cls._configs[category], updates)

    @classmethod
    def load_persistent(cls):
        """REPLACES load_settings() from utility.py"""
        config_path = os.path.join(BASE_DIR, 'data', 'persistent.json')
        try:
            with open(config_path, 'r') as f:
                user_config = json.load(f)
            for category, settings in user_config.items():
                cls.update(category, settings)
        except FileNotFoundError:
            print("No persistent config found, using defaults")
        except json.JSONDecodeError as e:
            print(f"Invalid config file: {str(e)}")

# DataClasses...
@dataclass
class VideoMetadata:
    path: str
    duration: float
    frame_count: int
    fps: float
    resolution: Tuple[int, int]
    filesize: int
    has_audio: bool

@dataclass
class SceneData:
    start_frame: int
    end_frame: int
    scene_type: str = ConfigManager.get('processing', 'scene_detection.default_scene_type', 'gameplay')
    motion_score: float = 0.0
    complexity: float = 0.0
    audio_activity: float = 0.0
    speed_factor: float = ConfigManager.get('speed', 'min_speed_factor')
    transitions: List[Dict[str, Any]] = None

@dataclass
class ProcessingState:
    stage: str
    progress: float
    current_frame: int
    total_frames: int
    processed_scenes: int
    total_scenes: int
    start_time: float
    estimated_completion: float

# Global State Management ------------------------------------------------------
class GlobalState:
    """Maintains runtime processing state"""
    def __init__(self):
        self.current_video: Optional[VideoMetadata] = None
        self.processing_state: Optional[ProcessingState] = None
        self.detected_scenes: List[SceneData] = []
        self.memory_usage: float = 0.0
        self.is_processing: bool = False
        self.cancel_requested: bool = False

    def reset(self):
        """Reset all state values"""
        self.__init__()

GLOBAL_STATE = GlobalState()

# Functions...
def get_full_path(path_key: str) -> str:
    """REPLACES direct PATHS dictionary access"""
    relative_path = ConfigManager.get('paths', path_key)
    return os.path.join(BASE_DIR, relative_path)

def update_processing_state(stage: str, progress: float, **kwargs):
    """Update global processing state with additional metrics"""
    if GLOBAL_STATE.processing_state:
        GLOBAL_STATE.processing_state.stage = stage
        GLOBAL_STATE.processing_state.progress = progress
        for key, value in kwargs.items():
            if hasattr(GLOBAL_STATE.processing_state, key):
                setattr(GLOBAL_STATE.processing_state, key, value)

# Initialize with persistent settings
ConfigManager.load_persistent()
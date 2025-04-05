# Script: `.\scripts\temporary.py`

# Imports...
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple
import os

# Globals...
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Maps etc...
PROCESSING_CONFIG = {
    # Video Settings
    'preview_height': 360,
    'target_height': 720,
    'frame_sample_rate': 30,
    'max_speed_factor': 4.0,
    'min_speed_factor': 1.0,
    'transition_frames': 30,
    'supported_formats': ['.mp4', '.avi', '.mkv'],
    'output_codec': 'libx264',
    'audio_codec': 'aac',
    
    # Scene Detection
    'static_threshold': 0.98,
    'menu_threshold': 0.85,
    'motion_threshold': 0.3,
    'texture_threshold': 0.4,
    'min_scene_duration': 2.0,
    'max_scene_duration': 300.0,
    'max_speed_factor': 4.0,
    'scene_similarity_threshold': 0.85,
    
    # Frame Processing
    'frame_settings': {
        'sample_rate': 30,
        'min_segment': 2,
        'max_segment': 30,
        'batch_size': 100
    },
    
    # Video Processing
    'video_settings': {
        'target_fps': 30,
        'min_clip_length': 2,
        'target_length': 30,  # minutes
        'resolution_height': 720,
        'preview_height': 360
    },
    
    # Hardware Acceleration
    'hardware_acceleration': {
        'use_opencl': True,  # Enable OpenCL by default for motion detection
        'use_avx2': False    # Disable AVX2 by default unless explicitly supported
    }
}

# Audio Processing Configuration
AUDIO_CONFIG = {
    'sample_rate': 44100,
    'window_size': 2048,
    'hop_length': 512,
    'threshold': 0.7,
    'preserve_pitch': True,
    'enhance_audio': True,
    
    # Audio Feature Detection
    'feature_settings': {
        'min_duration': 0.1,
        'max_duration': 30.0,
        'energy_threshold': 0.7,
        'frequency_threshold': 0.6
    }
}

# Memory Management Configuration
MEMORY_CONFIG = {
    'max_memory_usage': 0.8,
    'warning_threshold': 0.7,
    'critical_threshold': 0.9,
    'cleanup_interval': 60,  # seconds
    'frame_buffer_size': 30,
    'chunk_size_mb': 64
}

# Hardware Configuration
HARDWARE_CONFIG = {
    # Hardware capabilities (detected)
    'OpenCL': False,
    'AVX2': False,
    'AOCL': False,
    'x64': False,
    
    # User preferences (configurable)
    'use_gpu': True,
    'use_opencl': True,
    'use_avx2': True,
    
    # Performance settings
    'gpu_batch_size': 32,
    'cpu_threads': 4,
    
    # Platform preferences
    'opencl_platform_preference': ['NVIDIA', 'AMD', 'Intel']
}

# File System Paths
PATHS = {
    'input': 'input',
    'output': 'output',
    'work': 'data/temp',
    'data': 'data'
}

# Scene Settings
SCENE_CONFIG = {
    'min_scene_length': 2.0,    # seconds
    'max_scene_length': 300.0,  # seconds
    'scene_threshold': 30.0,    # threshold for scene change detection
    'action_threshold': 0.3,    # threshold for action sequence detection
    'similarity_threshold': 0.85,# threshold for content similarity
    'static_duration': 1.0,     # duration to keep from static scenes
    'menu_speed_factor': 2.0    # speed factor for menu scenes
}

# Speed Adjustment Settings
SPEED_CONFIG = {
    'max_speed_factor': 4.0,    # maximum speedup for non-action scenes
    'min_speed_factor': 1.0,    # minimum speedup (normal speed)
    'transition_frames': 30,    # frames for speed transition
    'smooth_window': 15,        # frames for smoothing speed changes
    'action_speed': 1.0,        # speed for action sequences
    'menu_speed': 2.0,         # speed for menu sequences
    'static_speed': 4.0        # speed for static sequences
}

# Analysis Settings
ANALYSIS_CONFIG = {
    'motion_threshold': 0.3,
    'texture_threshold': 0.4,
    'static_threshold': 0.95,
    'menu_threshold': 0.7,
    'action_threshold': 0.6,
    'min_scene_duration': 2.0,
    'frame_sample_rate': 5,
    'batch_size': 32
}

# Progress Tracking Configuration
PROGRESS_CONFIG = {
    'stages': [
        'initialization',
        'analysis',
        'processing',
        'compilation',
        'finalization'
    ],
    'weights': {
        'initialization': 0.05,
        'analysis': 0.2,
        'processing': 0.5,
        'compilation': 0.2,
        'finalization': 0.05
    }
}

# Error Handling Configuration
ERROR_CONFIG = {
    'max_retries': 3,
    'retry_delay': 5,
    'critical_errors': [
        'MemoryError',
        'IOError',
        'RuntimeError'
    ],
    'warning_thresholds': {
        'memory_usage': 0.8,
        'processing_time': 300,
        'file_size': 4_000_000_000
    }
}

@dataclass
class VideoMetadata:
    """Video file metadata."""
    path: str
    duration: float
    frame_count: int
    fps: float
    resolution: Tuple[int, int]
    filesize: int
    has_audio: bool

@dataclass
class SceneData:
    """Scene information for processing."""
    start_frame: int
    end_frame: int
    scene_type: str
    motion_score: float
    complexity: float
    audio_activity: float
    speed_factor: float
    transitions: List[Dict[str, Any]]

@dataclass
class ProcessingState:
    """Current state of video processing."""
    stage: str
    progress: float
    current_frame: int
    total_frames: int
    processed_scenes: int
    total_scenes: int
    start_time: float
    estimated_completion: float

# Global state (updated during processing)
class GlobalState:
    """Global processing state manager."""
    def __init__(self):
        self.current_video: Optional[VideoMetadata] = None
        self.processing_state: Optional[ProcessingState] = None
        self.detected_scenes: List[SceneData] = []
        self.memory_usage: float = 0.0
        self.is_processing: bool = False
        self.cancel_requested: bool = False

    def reset(self):
        """Reset global state."""
        self.current_video = None
        self.processing_state = None
        self.detected_scenes = []
        self.memory_usage = 0.0
        self.is_processing = False
        self.cancel_requested = False
GLOBAL_STATE = GlobalState()  #-- Do not move above GlobalState

# Functions...
def get_full_path(path_key: str) -> str:
    """Get full path for a path key relative to the project root."""
    return os.path.join(BASE_DIR, PATHS[path_key])

def update_processing_state(stage: str, progress: float, **kwargs) -> None:
    """Update global processing state."""
    if GLOBAL_STATE.processing_state:
        GLOBAL_STATE.processing_state.stage = stage
        GLOBAL_STATE.processing_state.progress = progress
        for key, value in kwargs.items():
            setattr(GLOBAL_STATE.processing_state, key, value)
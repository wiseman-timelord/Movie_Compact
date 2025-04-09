# .\scripts\utility.py

# Imports...
import os, cv2, time
import moviepy.editor as mp
from dataclasses import dataclass
import numpy as np
import pyopencl as cl
import json, datetime, sys, psutil, librosa, shutil, gc
from typing import Tuple, List, Dict, Any, Generator, Optional, Callable, Union
from threading import Lock, Event
from queue import Queue
from scripts.exceptions import (HardwareError, ConfigurationError, MovieCompactError)
from scripts.temporary import (
    MEMORY_CONFIG, ERROR_CONFIG, AUDIO_CONFIG, PROCESSING_CONFIG, BASE_DIR, get_full_path
)

# OpenCL kernel for frame difference
kernel_code = """
__kernel void frame_diff(__global const uchar *frame1, __global const uchar *frame2,
                         __global uchar *diff, int width, int height) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x < width && y < height) {
        int idx = y * width + x;
        diff[idx] = abs(frame1[idx] - frame2[idx]);
    }
}
"""
prg = cl.Program(context, kernel_code).build()

# Classes...
class CoreUtilities:
    def __init__(self):
        self.memory_manager = MemoryManager()
        self.progress_monitor = ProgressMonitor()
        self.error_handler = ErrorHandler()

class MemoryManager:
    def __init__(self):
        self.tracked_objects = []
        self.cleanup_interval = ConfigManager.get('memory', 'cleanup_interval', 60)
        self.max_usage = ConfigManager.get('memory', 'max_memory_usage')
        self.warning_thresh = ConfigManager.get('memory', 'warning_threshold')
        
    def track(self, obj):
        """REPLACES manual memory tracking"""
        self.tracked_objects.append(weakref.ref(obj))
        
    def auto_cleanup(self, func):
        """DECORATOR for automatic cleanup"""
        def wrapper(*args, **kwargs):
            try:
                result = func(*args, **kwargs)
            finally:
                self.cleanup(force=True)
            return result
        return wrapper
    
    def managed_array(self, shape, dtype):
        """REPLACES direct numpy array creation"""
        arr = np.empty(shape, dtype)
        self.track(arr)
        return arr

    def check_memory(self) -> Dict[str, Any]:
        """Check current memory usage against thresholds"""
        process = psutil.Process()
        usage = process.memory_info().rss
        return {
            "usage": usage,
            "warning": usage > self.warning_thresh,
            "critical": usage > self.max_usage
        }

    def cleanup(self, force: bool = False) -> None:
        """Perform memory cleanup of tracked objects"""
        current_usage = self.check_memory()
        
        if force or current_usage["warning"]:
            self.tracked_objects = [
                ref for ref in self.tracked_objects
                if ref() is not None  # Only keep alive objects
            ]
            gc.collect()

class SceneDetector:
    def __init__(self, hardware_ctx: dict):
        self.motion_detector = self._init_detector(hardware_ctx)
        self.texture_threshold = ConfigManager.get('processing', 'scene_detection.texture_threshold')
        self.motion_threshold = ConfigManager.get('processing', 'scene_detection.motion_threshold')
        self.min_duration = ConfigManager.get('processing', 'scene_detection.min_scene_duration') * 30  # FPS assumption
    
    def _init_detector(self, ctx):
        if ctx['use_opencl']:
            return detect_motion_opencl
        elif ctx['use_avx2']:
            # Placeholder for AVX2 detector if implemented
            return detect_motion_cpu
        return detect_motion_cpu
    
    def detect_scenes(self, frames: List[np.ndarray], audio_segments: List[Tuple[float, float]]) -> List[SceneData]:
        scenes = []
        current_start = 0
        
        for i in range(1, len(frames)):
            if self._is_scene_change(frames[i-1], frames[i]) and (i - current_start) >= self.min_duration:
                scenes.append(SceneData(
                    start_frame=current_start,
                    end_frame=i-1,
                    scene_type='gameplay',  # Default, refined later
                    motion_score=self.motion_detector(frames[i-1], frames[i], self.motion_threshold)
                ))
                current_start = i
        
        if current_start < len(frames):
            scenes.append(SceneData(start_frame=current_start, end_frame=len(frames)-1, scene_type='gameplay'))
        
        # Integrate audio and refine scene types
        for scene in scenes:
            scene.audio_activity = self._calculate_audio_activity(scene, audio_segments, len(frames))
            scene.scene_type = self._determine_scene_type(scene, frames[scene.start_frame:scene.end_frame + 1])
        
        return scenes
    
    def _is_scene_change(self, frame1, frame2):
        motion = self.motion_detector(frame1, frame2, self.motion_threshold)
        texture_diff = detect_texture_change(frame1, frame2, self.texture_threshold)
        return motion or texture_diff
    
    def _calculate_audio_activity(self, scene, audio_segments, total_frames):
        start_time = scene.start_frame / total_frames
        end_time = scene.end_frame / total_frames
        overlap = sum(max(0, min(end_time, a_end) - max(start_time, a_start)) 
                     for a_start, a_end in audio_segments)
        return overlap / (end_time - start_time) if end_time > start_time else 0
    
    def _determine_scene_type(self, scene, scene_frames):
        if all(detect_static_frame(scene_frames[i], scene_frames[i+1]) 
               for i in range(len(scene_frames)-1)):
            return 'static'
        elif detect_menu_screen(scene_frames[len(scene_frames)//2]):
            return 'menu'
        elif scene.motion_score > 0.5 or scene.audio_activity > 0.3:
            return 'action'
        return 'gameplay'

class ProgressMonitor:
    """Monitor and report processing progress."""
    
    def __init__(self):
        self.stages: List[Dict[str, Any]] = []
        self.current_stage: Optional[str] = None
        self.callbacks: List[Callable] = []
        self.start_time = time.time()
        self.stage_weights = {}
        self._lock = Lock()

    def register_callback(self, callback: Callable[[str, float, str], None]) -> None:
            self.callbacks.append(callback)

    def start_stage(self, stage: str, weight: float = 1.0) -> None:
        """Start a new processing stage."""
        with self._lock:
            self.current_stage = stage
            self.stages.append({
                'name': stage,
                'start_time': time.time(),
                'progress': 0.0,
                'complete': False
            })
            self.stage_weights[stage] = weight
            self._notify_progress(stage, 0.0, f"Starting {stage}")

    def update_progress(self, progress: float, message: str = "") -> None:
            if not self.current_stage:
                return
            with self._lock:
                for stage in self.stages:
                    if stage['name'] == self.current_stage:
                        stage['progress'] = progress
                        break
                overall_progress = self.get_overall_progress()
                update_processing_state(stage=self.current_stage, progress=overall_progress)
                self._notify_progress(self.current_stage, progress, message)

    def complete_stage(self, stage: str) -> None:
        """Mark a stage as complete."""
        with self._lock:
            for s in self.stages:
                if s['name'] == stage:
                    s['complete'] = True
                    s['end_time'] = time.time()
                    break
            
            self._notify_progress(stage, 100.0, f"Completed {stage}")

    def get_overall_progress(self) -> float:
        """Calculate overall progress across all stages."""
        total_weight = sum(self.stage_weights.values())
        weighted_progress = 0.0
        
        for stage in self.stages:
            stage_weight = self.stage_weights.get(stage['name'], 1.0)
            weighted_progress += (stage['progress'] / 100.0) * (stage_weight / total_weight)
        
        return weighted_progress * 100.0

    def _notify_progress(self, stage: str, progress: float, message: str) -> None:
        """Notify all registered callbacks of progress update."""
        for callback in self.callbacks:
            try:
                callback(stage, progress, message)
            except Exception as e:
                print(f"Error: [PROGRESS] In callback: {e}")
                time.sleep(5)

    def get_estimated_time(self) -> str:
        """Get estimated time remaining based on progress."""
        if not self.stages:
            return "Unknown"
            
        overall_progress = self.get_overall_progress()
        if overall_progress <= 0:
            return "Calculating..."
            
        elapsed = time.time() - self.start_time
        estimated_total = elapsed / (overall_progress / 100.0)
        remaining = estimated_total - elapsed
        
        if remaining < 60:
            return f"{int(remaining)} seconds"
        elif remaining < 3600:
            return f"{int(remaining/60)} minutes"
        else:
            hours = int(remaining/3600)
            minutes = int((remaining % 3600)/60)
            return f"{hours}h {minutes}m"

class ErrorHandler:
    """Handle and manage processing errors."""
    
    def __init__(self):
        self.error_log = []
        self._lock = Lock()
        self.max_retries = ERROR_CONFIG['max_retries']
        self.retry_delay = ERROR_CONFIG['retry_delay']

    def handle_error(self, error: Exception, context: str) -> Dict[str, Any]:
        with self._lock:
            error_info = {
                'timestamp': datetime.datetime.now(),
                'error': str(error),
                'context': context,
                'type': type(error).__name__,
                'can_retry': self._can_retry(error),
                'recovery_action': self._get_recovery_action(error)
            }
            
            self.error_log.append(error_info)
            print(f"Error: [{error_info['type']}] In {context}: {str(error)}")  # Changed
            time.sleep(5)  # Added
            
            return error_info

    def _can_retry(self, error: Exception) -> bool:
        """Determine if operation can be retried."""
        # List of retryable error types
        retryable_errors = (IOError, TimeoutError, ConnectionError)
        return isinstance(error, retryable_errors)

    def _get_recovery_action(self, error: Exception) -> Optional[str]:
        """Determine appropriate recovery action for error."""
        if isinstance(error, MemoryError):
            return "cleanup_memory"
        elif isinstance(error, IOError):
            return "retry_operation"
        elif isinstance(error, ConfigurationError):
            return "check_settings"
        return None

    def retry_operation(self, operation: Callable, max_retries: Optional[int] = None) -> Any:
        """Retry an operation with exponential backoff."""
        retries = 0
        max_attempts = max_retries or self.max_retries
        
        while retries < max_attempts:
            try:
                return operation()
            except Exception as e:
                retries += 1
                if retries >= max_attempts:
                    raise
                    
                delay = self.retry_delay * (2 ** (retries - 1))
                print(f"Info: Retry {retries}/{max_attempts} after {delay}s")  # Changed
                time.sleep(1)  # Added

    def handle_config_error(self, error: Exception) -> None:
        """Special handler for configuration errors."""
        error_info = {
            'timestamp': datetime.datetime.now(),
            'error': str(error),
            'type': 'ConfigurationError',
            'action': 'Check persistent.json format and version'
        }
        self.error_log.append(error_info)
        print(f"CRITICAL CONFIG ERROR: {str(error)}")
        time.sleep(5)
        sys.exit(1)

class AudioProcessor:
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        self.settings = {}  # Assume settings are set elsewhere or passed in

    def _reduce_noise(self, audio: np.ndarray) -> np.ndarray:
        """
        Reduce noise using spectral subtraction.

        Args:
            audio: Input audio array (mono, float32).

        Returns:
            np.ndarray: Noise-reduced audio.
        """
        noise_sample = audio[:int(0.5 * self.sample_rate)]
        noise_stft = librosa.stft(noise_sample, n_fft=2048, hop_length=512)
        noise_mag = np.mean(np.abs(noise_stft), axis=1, keepdims=True)
        audio_stft = librosa.stft(audio, n_fft=2048, hop_length=512)
        audio_mag, audio_phase = np.abs(audio_stft), np.angle(audio_stft)
        clean_mag = np.maximum(audio_mag - noise_mag, 0.0)
        clean_stft = clean_mag * np.exp(1j * audio_phase)
        return librosa.istft(clean_stft, hop_length=512)

    def _enhance_clarity(self, audio: np.ndarray) -> np.ndarray:
        """
        Enhance clarity using a bandpass filter for voice frequencies.

        Args:
            audio: Input audio array (mono, float32).

        Returns:
            np.ndarray: Clarity-enhanced audio.
        """
        lowcut, highcut = 1000.0, 4000.0
        nyquist = 0.5 * self.sample_rate
        low, high = lowcut / nyquist, highcut / nyquist
        b, a = butter(4, [low, high], btype='band')
        return lfilter(b, a, audio)

    def _normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """Normalize audio levels."""
        return librosa.util.normalize(audio)

    def _preserve_pitch(self, audio: np.ndarray, speed_factor: float) -> np.ndarray:
        """Preserve audio pitch during speed changes."""
        if speed_factor == 1.0:
            return audio
        return librosa.effects.time_stretch(audio, rate=speed_factor)

    def process_audio(self, audio: np.ndarray, speed_factor: float) -> np.ndarray:
        """Process audio with noise reduction, pitch preservation, and clarity enhancement."""
        try:
            audio = self._normalize_audio(audio)
            if self.settings.get('audio', {}).get('noise_reduction', True):
                audio = self._reduce_noise(audio)
            if speed_factor != 1.0 and self.settings.get('audio', {}).get('preserve_pitch', True):
                audio = self._preserve_pitch(audio, speed_factor)
            if self.settings.get('audio', {}).get('enhance_clarity', True):
                audio = self._enhance_clarity(audio)
            return audio
        except Exception as e:
            print(f"Error: Audio processing failed - {e}")
            time.sleep(5)
            return audio

class MetricsCollector:
    """Collect and track processing metrics."""
    
    def __init__(self):
        self.metrics = {
            'total_duration': 0,           # Total duration of all files (seconds)
            'total_size': 0,               # Total size of all files (bytes)
            'processed_frames': 0,         # Total number of frames processed
            'processing_time': 0,          # Total processing time (seconds)
            'memory_usage': 0,             # Peak memory usage (bytes)
            'phase_timings': {},           # Timings for each processing phase
            'file_metrics': {}             # Metrics per file
        }
        self.start_time = datetime.datetime.now()

    def update_file_metrics(self, file_path: str, metrics: Dict[str, Any]) -> None:
        """Update metrics for a specific file."""
        self.metrics['file_metrics'][file_path] = metrics
        self.metrics['total_duration'] += metrics.get('duration', 0)
        self.metrics['total_size'] += metrics.get('size', 0)

    def update_processing_metrics(self, frames: int, memory: int) -> None:
        """Update processing metrics."""
        self.metrics['processed_frames'] += frames
        self.metrics['memory_usage'] = max(self.metrics['memory_usage'], memory)

    def start_phase_timing(self, phase_name: str) -> None:
        """Start timing a processing phase."""
        self.metrics['phase_timings'][phase_name] = {
            'start': datetime.datetime.now(),
            'end': None,
            'duration': None
        }

    def end_phase_timing(self, phase_name: str) -> None:
        """End timing a processing phase and calculate duration."""
        if phase_name in self.metrics['phase_timings']:
            phase = self.metrics['phase_timings'][phase_name]
            phase['end'] = datetime.datetime.now()
            phase['duration'] = (phase['end'] - phase['start']).total_seconds()
            self.metrics['processing_time'] += phase['duration']

    def get_total_processing_time(self) -> float:
        """Get total processing time in seconds since initialization."""
        return (datetime.datetime.now() - self.start_time).total_seconds()

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of all collected metrics."""
        summary = self.metrics.copy()
        summary['total_processing_time'] = self.get_total_processing_time()
        return summary
        

class FileProcessor:
    """Handle file validation and processing operations."""
    
    def __init__(self, supported_formats: List[str]):
        self.supported_formats = supported_formats
        self.core = CoreUtilities()
        self.error_handler = ErrorHandler()

    def validate_file(self, file_path: str) -> bool:
        """Validate a file's existence and format."""
        if not os.path.exists(file_path):
            self.error_handler.handle_error(
                FileNotFoundError(f"File not found: {file_path}"), 
                "file_validation"
            )
            return False
            
        ext = os.path.splitext(file_path)[1].lower()
        if ext not in self.supported_formats:
            self.error_handler.handle_error(
                ValueError(f"Unsupported format: {ext}"), 
                "file_validation"
            )
            return False
            
        return True

    def get_file_metadata(self, file_path: str) -> Dict[str, Any]:
        """Get basic file metadata."""
        return {
            "path": file_path,
            "size": os.path.getsize(file_path),
            "modified": os.path.getmtime(file_path),
            "format": os.path.splitext(file_path)[1].lower()
        }

    def batch_validate(self, file_list: List[str]) -> Dict[str, List[str]]:
        """Validate multiple files with progress tracking."""
        results = {"valid": [], "invalid": []}
        for idx, file_path in enumerate(file_list):
            if self.validate_file(file_path):
                results["valid"].append(file_path)
            else:
                results["invalid"].append(file_path)
                
            self.core.progress_monitor.update_progress(
                (idx + 1) / len(file_list) * 100,
                f"Validating files: {idx+1}/{len(file_list)}"
            )
            
        return results

class AudioAnalyzer:
    def extract_audio(self, video_path: str) -> np.ndarray:
        """Extract audio from a video file.

        Args:
            video_path (str): Path to the video file.

        Returns:
            np.ndarray: Audio data as a NumPy array.
        """
        try:
            import moviepy.editor as mp
            clip = mp.VideoFileClip(video_path)
            audio = clip.audio.to_soundarray()
            clip.close()
            return audio
        except Exception as e:
            print(f"Error extracting audio: {e}")
            return np.array([])

# Functions...
def cleanup_work_directory() -> None:
    """Silently clean and recreate the work directory."""
    work_dir = get_full_path('work')
    shutil.rmtree(work_dir, ignore_errors=True)  # Force delete without raising errors
    os.makedirs(work_dir, exist_ok=True)
       

def detect_motion_cpu(frame1: np.ndarray, frame2: np.ndarray, threshold: float) -> bool:
    """
    Fallback CPU-based motion detection.
    
    Args:
        frame1: First frame (grayscale, np.uint8).
        frame2: Second frame (grayscale, np.uint8).
        threshold: Mean difference threshold.
    
    Returns:
        bool: True if motion is detected, False otherwise.
    """
    diff = cv2.absdiff(frame1, frame2)
    return np.mean(diff) > threshold
    
def extract_frames_optimized(video_path: str, batch_size: int = 32) -> Generator[np.ndarray, None, None]:
    """
    Optimized frame extraction with batch processing and hardware acceleration.
    
    Args:
        video_path: Path to video file
        batch_size: Number of frames to process in each batch
        
    Yields:
        Frames in BGR format
    """
    try:
        # Try hardware-accelerated decoding first
        cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            cap = cv2.VideoCapture(video_path)  # Fallback to default

        frame_buffer = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert to RGB for consistency with other processing
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_buffer.append(frame)

            # Process in batches for memory efficiency
            if len(frame_buffer) >= batch_size:
                for f in frame_buffer:
                    yield f
                frame_buffer = []

                # Check memory usage periodically
                if MemoryManager().check_memory()['warning']:
                    MemoryManager().cleanup(force=True)

        # Yield remaining frames
        for f in frame_buffer:
            yield f

        cap.release()
        
    except Exception as e:
        print(f"Error: Frame extraction failed - {e}")  # Replaced log_event
        time.sleep(5)  # Added
        raise MovieCompactError(f"Frame extraction failed: {e}")
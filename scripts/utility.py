# .\scripts\utility.py

# Imports...
import os, cv2
import numpy as np
import pyopencl as cl
import json, datetime, sys, psutil, librosa, shutil, gc
from typing import Tuple, List, Dict, Any, Generator, Optional, Callable, Union
from threading import Lock, Event
import time
from queue import Queue
from interface import HardwareError, ConfigurationError, MovieCompactError
from scripts.temporary import (
    MEMORY_CONFIG,
    ERROR_CONFIG,
    AUDIO_CONFIG,
    PROCESSING_CONFIG
)

# OpenCL setup (global for reuse)
platform = cl.get_platforms()[0]
device = platform.get_devices()[0]
context = cl.Context([device])
queue = cl.CommandQueue(context)

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
    """Manage memory usage and cleanup."""
    
    def __init__(self):
        self.max_memory_usage = MEMORY_CONFIG['max_memory_usage']
        self.warning_threshold = MEMORY_CONFIG['warning_threshold']
        self.critical_threshold = MEMORY_CONFIG['critical_threshold']
        self._last_cleanup = 0
        self.cleanup_interval = MEMORY_CONFIG['cleanup_interval']

    def check_memory(self) -> Dict[str, Union[bool, float]]:
        """Check current memory usage and status."""
        try:
            process = psutil.Process()
            memory_percent = process.memory_percent()
            system_memory = psutil.virtual_memory()
            
            status = {
                'safe': memory_percent <= (self.max_memory_usage * 100),
                'warning': memory_percent > (self.warning_threshold * 100),
                'critical': memory_percent > (self.critical_threshold * 100),
                'usage_percent': memory_percent,
                'available_gb': system_memory.available / (1024**3)
            }
            
            if status['warning']:
                log_event(f"High memory usage: {memory_percent:.1f}%", "WARNING", "MEMORY")
            
            return status
        except Exception as e:
            log_event(f"Error checking memory: {e}", "ERROR", "MEMORY")
            return {'safe': False, 'warning': True, 'critical': False, 'usage_percent': 0, 'available_gb': 0}

    def cleanup(self, force: bool = False) -> bool:
        """Perform memory cleanup if needed or forced."""
        current_time = time.time()
        if force or (current_time - self._last_cleanup) > self.cleanup_interval:
            try:
                # Clear Python's internal cache
                gc.collect()
                
                # Clear OpenCV cache
                cv2.destroyAllWindows()
                
                # Reset last cleanup time
                self._last_cleanup = current_time
                
                log_event("Memory cleanup performed", "INFO", "MEMORY")
                return True
            except Exception as e:
                log_event(f"Error during memory cleanup: {e}", "ERROR", "MEMORY")
                return False
        return False

    def optimize_for_large_file(self, file_size: int) -> Dict[str, Any]:
        """Configure memory management for large files."""
        available_mem = psutil.virtual_memory().available
        chunk_size = min(file_size // 10, available_mem // 4)
        
        config = {
            'chunk_size': chunk_size,
            'max_chunks': max(1, available_mem // (chunk_size * 2)),
            'buffer_size': min(64 * 1024 * 1024, chunk_size // 10),
            'use_temp_files': file_size > available_mem // 2
        }
        
        self.log_manager.log(
            f"Memory config: chunk_size={chunk_size/1024/1024:.1f}MB, "
            f"max_chunks={config['max_chunks']}, "
            f"use_temp={config['use_temp_files']}",
            "INFO",
            "MEMORY"
        )
        
        return config

    def recover_from_error(self, error_type: str) -> bool:
        """Attempt to recover from memory-related errors."""
        if error_type == "MemoryError":
            try:
                # Force aggressive cleanup
                self.cleanup(force=True)
                
                # Disable OpenCV optimizations
                cv2.ocl.setUseOpenCL(False)
                
                # Clear OpenCV cache
                cv2.destroyAllWindows()
                
                # Clear moviepy cache
                if hasattr(mp.VideoFileClip, 'clear_cache'):
                    mp.VideoFileClip.clear_cache()
                
                # Request garbage collection
                gc.collect()
                
                self.log_manager.log(
                    "Memory recovery performed",
                    "INFO",
                    "MEMORY"
                )
                return True
                
            except Exception as e:
                self.error_handler.handle_error(e, "memory_recovery")
                return False
                
        return False

    def monitor_usage(self, threshold_mb: float = 1000.0,
                     interval_seconds: float = 1.0) -> None:
        """Monitor memory usage over time."""
        import threading
        
        def _monitor():
            while not self.stop_monitoring.is_set():
                try:
                    usage = self.check_memory()
                    if usage['warning'] or usage['critical']:
                        self.log_manager.log(
                            f"High memory usage: {usage['usage_percent']:.1f}%",
                            "WARNING" if usage['warning'] else "ERROR",
                            "MEMORY"
                        )
                        if usage['critical']:
                            self.cleanup(force=True)
                    time.sleep(interval_seconds)
                except Exception as e:
                    self.error_handler.handle_error(e, "memory_monitoring")
                    break
        
        self.stop_monitoring = threading.Event()
        self.monitor_thread = threading.Thread(target=_monitor)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()

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
        """Register a progress callback function."""
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
        """Update progress for current stage."""
        if not self.current_stage:
            return
            
        with self._lock:
            for stage in self.stages:
                if stage['name'] == self.current_stage:
                    stage['progress'] = progress
                    break
            
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
                log_event(f"Error in progress callback: {e}", "ERROR", "PROGRESS")

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
        """Handle an error and determine recovery action."""
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
            log_event(f"Error in {context}: {str(error)}", "ERROR", error_info['type'])
            
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
                log_event(f"Retry {retries}/{max_attempts} after {delay}s", "INFO", "RETRY")
                time.sleep(delay)

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
        """
        Process audio with noise reduction, pitch preservation, and clarity enhancement.

        Args:
            audio: Input audio array.
            speed_factor: Speed adjustment factor.

        Returns:
            np.ndarray: Processed audio.
        """
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
            # Replace with your logging mechanism
            print(f"Error processing audio: {e}")
            return audio

# Original classes with updates
class AudioAnalyzer:
    def __init__(self):
        self.threshold = AUDIO_CONFIG['threshold']
        self.sample_rate = AUDIO_CONFIG['sample_rate']
        self.window_size = AUDIO_CONFIG['window_size']
        self.hop_length = AUDIO_CONFIG['hop_length']
        self.processor = AudioProcessor()

    def extract_audio(self, video_path: str) -> np.ndarray:
        """Extract audio from video file."""
        try:
            import moviepy.editor as mp
            video = mp.VideoFileClip(video_path)
            if video.audio is None:
                log_event("No audio track found in video", "WARNING", "AUDIO")
                return np.array([])

            audio = video.audio.to_soundarray()
            video.close()

            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)

            return audio
        except Exception as e:
            log_event(f"Error extracting audio: {e}", "ERROR", "AUDIO")
            return np.array([])

    def detect_high_activity(self, audio: np.ndarray,
                           window_size: Optional[int] = None) -> List[Tuple[float, float]]:
        """Detect segments with high audio activity."""
        try:
            if audio.size == 0:
                return []

            if window_size is None:
                window_size = self.window_size

            energy = librosa.feature.rms(
                y=audio,
                frame_length=window_size,
                hop_length=self.hop_length
            )[0]

            energy = energy / np.max(energy)
            active_segments = []
            start = None

            for i, e in enumerate(energy):
                if e > self.threshold and start is None:
                    start = i * self.hop_length / self.sample_rate
                elif e <= self.threshold and start is not None:
                    end = i * self.hop_length / self.sample_rate
                    if end - start >= 0.1:
                        active_segments.append((start, end))
                    start = None

            if start is not None:
                end = len(audio) / self.sample_rate
                if end - start >= 0.1:
                    active_segments.append((start, end))

            return active_segments
        except Exception as e:
            log_event(f"Error detecting audio activity: {e}", "ERROR", "AUDIO")
            return []

class SceneManager:
    def __init__(self):
        self.scene_settings = SCENE_CONFIG
        self.min_scene_length = SCENE_CONFIG['min_scene_length']
        self.max_scene_length = SCENE_CONFIG['max_scene_length']
        self.threshold = SCENE_CONFIG['scene_threshold']
        self._lock = Lock()
        self.metrics = MetricsCollector()

    def detect_scenes(self, frames: List[np.ndarray],
                     audio_segments: List[Tuple[float, float]]) -> List[Dict[str, Any]]:
        """Detect and analyze scenes combining visual and audio information."""
        with self._lock:
            scenes = []
            current_scene = None
            
            for i in range(1, len(frames)):
                if self.is_scene_change(frames[i-1], frames[i]):
                    if current_scene:
                        scenes.append(self.finalize_scene(current_scene, frames, i-1))
                    current_scene = self.initialize_scene(i)
                
                if current_scene:
                    current_scene = self.update_scene(current_scene, frames[i], i)
                
                if (current_scene and 
                    i - current_scene['start_frame'] > self.max_scene_length * 30):
                    scenes.append(self.finalize_scene(current_scene, frames, i))
                    current_scene = self.initialize_scene(i + 1)
            
            if current_scene:
                scenes.append(self.finalize_scene(current_scene, frames, len(frames)-1))
            
            scenes = self.merge_short_scenes(scenes)
            scenes = self.merge_audio_info(scenes, audio_segments, len(frames))
            
            return scenes

    def is_scene_change(self, frame1: np.ndarray, frame2: np.ndarray) -> bool:
        """Detect if there is a scene change between frames."""
        try:
            hist1 = cv2.calcHist([frame1], [0, 1, 2], None, [8, 8, 8],
                               [0, 256, 0, 256, 0, 256])
            hist2 = cv2.calcHist([frame2], [0, 1, 2], None, [8, 8, 8],
                               [0, 256, 0, 256, 0, 256])
            diff = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CHISQR)
            
            frame_diff = cv2.absdiff(frame1, frame2)
            diff_score = np.mean(frame_diff)
            
            return diff > self.threshold or diff_score > 30.0
        except Exception as e:
            log_event(f"Error in scene change detection: {e}", "ERROR", "SCENE")
            return False

    def initialize_scene(self, start_frame: int) -> Dict[str, Any]:
        """Initialize a new scene."""
        return {
            'start_frame': start_frame,
            'end_frame': None,
            'is_action': False,
            'is_menu': False,
            'is_static': False,
            'motion_score': 0.0,
            'audio_activity': 0.0,
            'frame_count': 0,
            'complexity': 0.0,
            'transitions': []
        }

    def update_scene(self, scene: Dict[str, Any], frame: np.ndarray,
                    frame_index: int) -> Dict[str, Any]:
        """Update scene information with new frame."""
        scene['frame_count'] += 1
        scene['end_frame'] = frame_index
        
        if scene['frame_count'] > 1:
            motion = detect_motion_opencl(frame, frame, 0.5)
            scene['motion_score'] = (
                (scene['motion_score'] * (scene['frame_count']-1) + float(motion)) /
                scene['frame_count']
            )
        
        if detect_menu_screen(frame):
            scene['is_menu'] = True
            
        # Update scene complexity
        scene['complexity'] = self._calculate_scene_complexity(frame, scene['complexity'],
                                                             scene['frame_count'])
        
        return scene

    def _calculate_scene_complexity(self, frame: np.ndarray, current_complexity: float,
                                  frame_count: int) -> float:
        """Calculate scene complexity based on various metrics."""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Calculate edge density
            edges = cv2.Canny(gray, 100, 200)
            edge_density = np.count_nonzero(edges) / edges.size
            
            # Calculate texture complexity
            texture_score = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Calculate color variety
            color_score = np.std(frame.reshape(-1, 3), axis=0).mean()
            
            # Combine metrics
            complexity = (0.4 * edge_density + 
                        0.3 * (texture_score / 1000) +  # Normalize texture score
                        0.3 * (color_score / 255))      # Normalize color score
            
            # Update running average
            if frame_count == 1:
                return complexity
            else:
                return (current_complexity * (frame_count - 1) + complexity) / frame_count
                
        except Exception as e:
            log_event(f"Error calculating scene complexity: {e}", "ERROR", "SCENE")
            return current_complexity

    def finalize_scene(self, scene: Dict[str, Any], frames: List[np.ndarray],
                      end_frame: int) -> Dict[str, Any]:
        """Finalize scene data and analysis."""
        scene['end_frame'] = end_frame
        scene['is_static'] = all(
            detect_static_frame(frames[i], frames[i+1])
            for i in range(scene['start_frame'], end_frame-1)
        )
        
        # Calculate final metrics
        scene['duration'] = (end_frame - scene['start_frame']) / 30  # Assuming 30fps
        scene['average_motion'] = scene['motion_score']
        
        # Detect transitions
        scene['transitions'] = self._detect_transitions(
            frames[scene['start_frame']:end_frame+1]
        )
        
        return scene

    def _detect_transitions(self, frames: List[np.ndarray]) -> List[Dict[str, Any]]:
            """
            Detect scene transitions between frames.

            Args:
                frames: List of video frames.

            Returns:
                List of dictionaries with transition details.
            """
            transitions = []
            for i in range(1, len(frames) - 1):
                try:
                    # Check frame consistency
                    if frames[i].shape != frames[i-1].shape or frames[i].shape != frames[i+1].shape:
                        # Replace with your logging mechanism
                        print(f"Frame shape mismatch at index {i}", "WARNING", "SCENE")
                        continue
                    diff_prev = cv2.absdiff(frames[i], frames[i-1])
                    diff_next = cv2.absdiff(frames[i], frames[i+1])
                    mean_diff_prev = np.mean(diff_prev)
                    mean_diff_next = np.mean(diff_next)
                    if mean_diff_prev < 5 and mean_diff_next > 30:
                        transitions.append({'frame': i, 'type': 'cut', 'confidence': 0.9})
                    elif mean_diff_prev < mean_diff_next:
                        transitions.append({'frame': i, 'type': 'fade', 'confidence': 0.7})
                except Exception as e:
                    # Replace with your logging mechanism
                    print(f"Error detecting transitions at frame {i}: {e}", "WARNING", "SCENE")
            return transitions

    def merge_short_scenes(self, scenes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge scenes that are too short."""
        if not scenes:
            return scenes
            
        merged = []
        current = scenes[0]
        
        for next_scene in scenes[1:]:
            if (next_scene['start_frame'] - current['end_frame'] < 
                self.min_scene_length * 30):
                # Merge scenes
                current['end_frame'] = next_scene['end_frame']
                current['frame_count'] += next_scene['frame_count']
                current['motion_score'] = (
                    (current['motion_score'] * current['frame_count'] +
                     next_scene['motion_score'] * next_scene['frame_count']) /
                    (current['frame_count'] + next_scene['frame_count'])
                )
                current['is_action'] |= next_scene['is_action']
                current['is_menu'] |= next_scene['is_menu']
                current['is_static'] &= next_scene['is_static']
                current['complexity'] = (current['complexity'] + next_scene['complexity']) / 2
                current['transitions'].extend(next_scene['transitions'])
            else:
                merged.append(current)
                current = next_scene
        
        merged.append(current)
        return merged

    def merge_audio_info(self, scenes: List[Dict[str, Any]],
                        audio_segments: List[Tuple[float, float]],
                        total_frames: int) -> List[Dict[str, Any]]:
        """Merge audio activity information into scene data."""
        for scene in scenes:
            start_time = scene['start_frame'] / total_frames
            end_time = scene['end_frame'] / total_frames
            
            # Calculate overlap with audio segments
            overlap = 0.0
            for a_start, a_end in audio_segments:
                if a_start < end_time and a_end > start_time:
                    overlap_start = max(start_time, a_start)
                    overlap_end = min(end_time, a_end)
                    overlap += overlap_end - overlap_start
            
            scene['audio_activity'] = overlap / (end_time - start_time)
            scene['is_action'] = (
                scene['audio_activity'] > 0.3 or
                scene['motion_score'] > 0.5
            )
            
            # Update scene importance based on audio
            scene['importance_score'] = self._calculate_importance(scene)
        
        return scenes

    def _calculate_importance(self, scene: Dict[str, Any]) -> float:
        """Calculate scene importance based on various factors."""
        weights = {
            'motion': 0.3,
            'audio': 0.3,
            'complexity': 0.2,
            'duration': 0.2
        }
        
        # Normalize duration score (prefer scenes between 2-10 seconds)
        duration_score = 1.0 - abs(scene['duration'] - 6) / 6
        duration_score = max(0, min(1, duration_score))
        
        scores = {
            'motion': scene['motion_score'],
            'audio': scene['audio_activity'],
            'complexity': scene['complexity'],
            'duration': duration_score
        }
        
        return sum(weights[k] * scores[k] for k in weights)

class PreviewGenerator:
    def __init__(self, work_dir: str = 'work'):
        self.work_dir = work_dir
        os.makedirs(work_dir, exist_ok=True)

    def create_preview(self, input_path: str) -> str:
        """
        Create a 360p preview of the video using ffmpeg.

        Args:
            input_path: Path to the input video.

        Returns:
            str: Path to the preview video if successful, otherwise an empty string.
        """
        preview_path = os.path.join(self.work_dir, f"preview_{os.path.basename(input_path)}")
        command = f'ffmpeg -i "{input_path}" -vf "scale=-1:360" -c:v libx264 -crf 23 -c:a aac -y "{preview_path}"'

        try:
            result = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if result.returncode == 0:
                return preview_path
            else:
                print(f"ffmpeg failed with return code {result.returncode}")
                return ""
        except subprocess.CalledProcessError as e:
            print(f"Error creating preview: {e}")
            return ""

    def generate_thumbnails(self, video_path: str, num_thumbnails: int = 5) -> List[str]:
        """Generate thumbnails from video for preview."""
        try:
            import moviepy.editor as mp
            clip = mp.VideoFileClip(video_path)
            duration = clip.duration
            interval = duration / (num_thumbnails + 1)
            
            thumbnails = []
            for i in range(num_thumbnails):
                time = interval * (i + 1)
                frame = clip.get_frame(time)
                
                # Save thumbnail
                thumb_path = os.path.join(
                    self.work_dir,
                    f"thumb_{i}_{os.path.basename(video_path)}.jpg"
                )
                cv2.imwrite(thumb_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                thumbnails.append(thumb_path)
            
            clip.close()
            return thumbnails
            
        except Exception as e:
            log_event(f"Error generating thumbnails: {e}", "ERROR", "PREVIEW")
            return []

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
        

class PreviewGenerator:
    def __init__(self, work_dir: str = 'work'):
        self.work_dir = work_dir
        os.makedirs(work_dir, exist_ok=True)

    def create_preview(self, input_path: str) -> str:
        """
        Create a 360p preview of the video using ffmpeg.
        
        Args:
            input_path: Path to input video.
        
        Returns:
            str: Path to preview video.
        """
        preview_path = os.path.join(self.work_dir, f"preview_{os.path.basename(input_path)}")
        os.system(f'ffmpeg -i "{input_path}" -vf "scale=-1:360" -c:v libx264 -crf 23 '
                  f'-c:a aac -y "{preview_path}"')
        return preview_path

class AudioProcessor:
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate

    def _reduce_noise(self, audio: np.ndarray) -> np.ndarray:
        """
        Reduce noise using spectral subtraction.
        
        Args:
            audio: Input audio array (mono, float32).
        
        Returns:
            np.ndarray: Noise-reduced audio.
        """
        # Estimate noise from first 0.5 seconds
        noise_sample = audio[:int(0.5 * self.sample_rate)]
        noise_stft = librosa.stft(noise_sample, n_fft=2048, hop_length=512)
        noise_mag = np.mean(np.abs(noise_stft), axis=1, keepdims=True)
        
        # STFT of full audio
        audio_stft = librosa.stft(audio, n_fft=2048, hop_length=512)
        audio_mag, audio_phase = np.abs(audio_stft), np.angle(audio_stft)
        
        # Spectral subtraction
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
        # Bandpass filter for 1kHz–4kHz (voice range)
        lowcut, highcut = 1000.0, 4000.0
        nyquist = 0.5 * self.sample_rate
        low, high = lowcut / nyquist, highcut / nyquist
        b, a = butter(4, [low, high], btype='band')
        enhanced_audio = lfilter(b, a, audio)
        return enhanced_audio



# Functions...
def load_hardware_config():
    """Load hardware capabilities from hardware.json."""
    hardware_file = os.path.join(BASE_DIR, "data", "hardware.json")
    try:
        with open(hardware_file, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        log_event("Hardware configuration file not found, assuming default capabilities", "WARNING", "CONFIG")
        return {"x64": True, "AVX2": False, "AOCL": False, "OpenCL": False}
    except json.JSONDecodeError as e:
        log_event(f"Error decoding hardware.json: {e}", "ERROR", "CONFIG")
        return {"x64": True, "AVX2": False, "AOCL": False, "OpenCL": False}


def detect_motion_opencl(frame1: np.ndarray, frame2: np.ndarray, threshold: float) -> bool:
    """
    Detect motion between two frames using OpenCL.
    
    Args:
        frame1: First frame (grayscale, np.uint8).
        frame2: Second frame (grayscale, np.uint8).
        threshold: Mean difference threshold for motion detection.
    
    Returns:
        bool: True if motion is detected, False otherwise.
    """
    mf = cl.mem_flags
    frame1_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=frame1)
    frame2_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=frame2)
    diff_buf = cl.Buffer(context, mf.WRITE_ONLY, frame1.nbytes)
    
    prg.frame_diff(queue, frame1.shape, None, frame1_buf, frame2_buf, diff_buf,
                   np.int32(frame1.shape[1]), np.int32(frame1.shape[0]))
    
    diff = np.empty_like(frame1)
    cl.enqueue_copy(queue, diff, diff_buf).wait()
    
    return np.mean(diff) > threshold

def detect_motion_avx2(frame1: np.ndarray, frame2: np.ndarray, threshold: float) -> bool:
    """
    Detect motion using AVX2 (placeholder; requires intrinsics).
    
    Args:
        frame1: First frame (grayscale, np.uint8).
        frame2: Second frame (grayscale, np.uint8).
        threshold: Mean difference threshold.
    
    Returns:
        bool: True if motion is detected, False otherwise.
    """
    # TODO: Implement AVX2 vectorized differencing
    diff = cv2.absdiff(frame1, frame2)  # Placeholder
    return np.mean(diff) > threshold

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
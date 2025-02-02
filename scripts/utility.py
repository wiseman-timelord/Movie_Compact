# utility.py

import os
import cv2
import numpy as np
import pyopencl as cl
import json
import datetime
import sys
import psutil
import librosa
import shutil
from typing import Tuple, List, Dict, Any, Generator, Optional
import moviepy.editor as mp
from threading import Lock
import time
from interface import HardwareError, ConfigurationError, MovieCompactError

class AudioAnalyzer:
    """Handle audio analysis operations for scene detection."""
    def __init__(self):
        self.settings = load_settings()
        self.threshold = self.settings.get('audio_threshold', 0.7)
        self.sample_rate = self.settings.get('audio', {}).get('sample_rate', 44100)
        self.window_size = self.settings.get('audio', {}).get('window_size', 2048)
        self.hop_length = self.settings.get('audio', {}).get('hop_length', 512)

    def extract_audio(self, video_path: str) -> np.ndarray:
        """Extract audio from video file."""
        try:
            # Use moviepy to extract audio
            video = mp.VideoFileClip(video_path)
            if video.audio is None:
                log_event("No audio track found in video", "WARNING", "AUDIO")
                return np.array([])

            # Extract audio as numpy array
            audio = video.audio.to_soundarray()
            video.close()

            # Convert to mono if stereo
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

            # Calculate short-time energy
            energy = librosa.feature.rms(
                y=audio,
                frame_length=window_size,
                hop_length=self.hop_length
            )[0]

            # Normalize energy
            energy = energy / np.max(energy)

            # Find segments above threshold
            active_segments = []
            start = None
            for i, e in enumerate(energy):
                if e > self.threshold and start is None:
                    start = i * self.hop_length / self.sample_rate
                elif e <= self.threshold and start is not None:
                    end = i * self.hop_length / self.sample_rate
                    if end - start >= 0.1:  # Minimum segment duration
                        active_segments.append((start, end))
                    start = None

            # Handle final segment
            if start is not None:
                end = len(audio) / self.sample_rate
                if end - start >= 0.1:
                    active_segments.append((start, end))

            return active_segments
        except Exception as e:
            log_event(f"Error detecting audio activity: {e}", "ERROR", "AUDIO")
            return []

    def analyze_audio_features(self, audio: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract detailed audio features for analysis."""
        try:
            features = {}
            
            # Spectral centroid
            features['spectral_centroid'] = librosa.feature.spectral_centroid(
                y=audio,
                sr=self.sample_rate,
                hop_length=self.hop_length
            )[0]

            # Spectral bandwidth
            features['spectral_bandwidth'] = librosa.feature.spectral_bandwidth(
                y=audio,
                sr=self.sample_rate,
                hop_length=self.hop_length
            )[0]

            # Root mean square energy
            features['rmse'] = librosa.feature.rms(
                y=audio,
                hop_length=self.hop_length
            )[0]

            # Zero crossing rate
            features['zero_crossing_rate'] = librosa.feature.zero_crossing_rate(
                y=audio,
                hop_length=self.hop_length
            )[0]

            return features
        except Exception as e:
            log_event(f"Error analyzing audio features: {e}", "ERROR", "AUDIO")
            return {}

class SceneManager:
    """Manage scene detection and analysis."""
    def __init__(self):
        self.settings = load_settings()
        self.scene_settings = self.settings.get('scene_settings', {})
        self.min_scene_length = self.scene_settings.get('min_scene_length', 2)
        self.max_scene_length = self.scene_settings.get('max_scene_length', 300)
        self.threshold = self.scene_settings.get('threshold', 30.0)

    def detect_scenes(self, frames: List[np.ndarray],
                     audio_segments: List[Tuple[float, float]]) -> List[Dict[str, Any]]:
        """Detect and analyze scenes combining visual and audio information."""
        scenes = []
        current_scene = None

        for i in range(1, len(frames)):
            # Check for scene change
            if self.is_scene_change(frames[i-1], frames[i]):
                if current_scene:
                    scenes.append(self.finalize_scene(current_scene, frames, i-1))
                current_scene = self.initialize_scene(i)

            # Update current scene
            if current_scene:
                current_scene = self.update_scene(current_scene, frames[i], i)

            # Force scene break if current scene is too long
            if (current_scene and 
                i - current_scene['start_frame'] > self.max_scene_length * 30):  # Assuming 30fps
                scenes.append(self.finalize_scene(current_scene, frames, i))
                current_scene = self.initialize_scene(i + 1)

        # Add final scene
        if current_scene:
            scenes.append(self.finalize_scene(current_scene, frames, len(frames)-1))

        # Merge short scenes
        scenes = self.merge_short_scenes(scenes)

        # Merge with audio information
        scenes = self.merge_audio_info(scenes, audio_segments, len(frames))

        return scenes

    def is_scene_change(self, frame1: np.ndarray, frame2: np.ndarray) -> bool:
        """Detect if there is a scene change between frames."""
        try:
            # Calculate color histogram difference
            hist1 = cv2.calcHist([frame1], [0, 1, 2], None, [8, 8, 8],
                               [0, 256, 0, 256, 0, 256])
            hist2 = cv2.calcHist([frame2], [0, 1, 2], None, [8, 8, 8],
                               [0, 256, 0, 256, 0, 256])
            diff = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CHISQR)

            # Calculate frame difference
            frame_diff = cv2.absdiff(frame1, frame2)
            diff_score = np.mean(frame_diff)

            # Combined decision
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
            'frame_count': 0
        }

    def update_scene(self, scene: Dict[str, Any], frame: np.ndarray,
                    frame_index: int) -> Dict[str, Any]:
        """Update scene information with new frame."""
        scene['frame_count'] += 1
        scene['end_frame'] = frame_index

        # Update motion score
        if scene['frame_count'] > 1:
            motion = detect_motion_opencl(frame, frame, 0.5)
            scene['motion_score'] = (
                (scene['motion_score'] * (scene['frame_count']-1) + float(motion)) /
                scene['frame_count']
            )

        # Check for menu screens
        if detect_menu_screen(frame):
            scene['is_menu'] = True

        return scene

    def finalize_scene(self, scene: Dict[str, Any], frames: List[np.ndarray],
                      end_frame: int) -> Dict[str, Any]:
        """Finalize scene data and analysis."""
        scene['end_frame'] = end_frame
        scene['is_static'] = all(
            detect_static_frame(frames[i], frames[i+1])
            for i in range(scene['start_frame'], end_frame-1)
        )
        return scene

    def merge_short_scenes(self, scenes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge scenes that are too short."""
        if not scenes:
            return scenes

        merged = []
        current = scenes[0]

        for next_scene in scenes[1:]:
            if (next_scene['start_frame'] - current['end_frame'] < 
                self.min_scene_length * 30):  # Assuming 30fps
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

        return scenes

class PreviewGenerator:
    """Handle preview video generation and management."""
    def __init__(self):
        self.settings = load_settings()
        self.preview_height = self.settings.get('video_settings', {}).get('preview_height', 360)
        self.work_dir = self.settings.get('video', {}).get('temp_directory', 'work')

    def create_preview(self, input_path: str) -> str:
        """Create lower resolution preview of video."""
        preview_path = os.path.join(self.work_dir, f"preview_{os.path.basename(input_path)}")

        try:
            clip = mp.VideoFileClip(input_path)
            aspect_ratio = clip.w / clip.h
            new_width = int(self.preview_height * aspect_ratio)

            preview = clip.resize(height=self.preview_height, width=new_width)
            preview.write_videofile(
                preview_path,
                codec='libx264',
                audio_codec='aac',
                preset='ultrafast',
                threads=4
            )

            clip.close()
            preview.close()

            log_event(f"Created preview: {preview_path}", "INFO", "PREVIEW")
            return preview_path

        except Exception as e:
            log_event(f"Error creating preview: {e}", "ERROR", "PREVIEW")
            return ""

class MetricsCollector:
    """Collect and track processing metrics."""
    def __init__(self):
        self.metrics = {
            'total_duration': 0,
            'total_size': 0,
            'processed_frames': 0,
            'processing_time': 0,
            'memory_usage': 0,
            'phase_timings': {},
            'file_metrics': {}
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
        """End timing a processing phase."""
        if phase_name in self.metrics['phase_timings']:
            phase = self.metrics['phase_timings'][phase_name]
            phase['end'] = datetime.datetime.now()
            phase['duration'] = (phase['end'] - phase['start']).total_seconds()

    def get_metrics_report(self) -> str:
        """Generate a detailed metrics report."""
        report = []
        report.append("Processing Metrics Report")
        report.append("-" * 50)

        # Overall metrics
        total_time = (datetime.datetime.now() - self.start_time).total_seconds()
        report.append(f"Total Processing Time: {total_time:.2f} seconds")
        report.append(
            f"Total Duration Processed: {self.metrics['total_duration']:.2f} seconds")
        report.append(
            f"Total Size Processed: {self.metrics['total_size'] / (1024*1024):.2f} MB")
        report.append(f"Total Frames Processed: {self.metrics['processed_frames']}")
        report.append(
            f"Peak Memory Usage: {self.metrics['memory_usage'] / (1024*1024):.2f} MB")

        # Phase timings
        report.append("\nPhase Timings:")
        for phase, timing in self.metrics['phase_timings'].items():
            if timing['duration'] is not None:
                report.append(f"  {phase}: {timing['duration']:.2f} seconds")

        # File metrics
        report.append("\nFile Details:")
        for file_path, metrics in self.metrics['file_metrics'].items():
            report.append(f"\n  {os.path.basename(file_path)}:")
            report.append(f"    Duration: {metrics.get('duration', 0):.2f} seconds")
            report.append(
                f"    Size: {metrics.get('size', 0) / (1024*1024):.2f} MB")
            report.append(f"    Frames: {metrics.get('frames', 0)}")

        # Performance metrics
        if total_time > 0:
            fps = self.metrics['processed_frames'] / total_time
            report.append(f"\nProcessing Performance:")
            report.append(f"  Average FPS: {fps:.2f}")
            report.append(
                f"  Processing Ratio: {self.metrics['total_duration']/total_time:.2f}x")

        return "\n".join(report)

class FileProcessor:
    """Handle file operations and tracking."""
    def __init__(self, supported_formats: List[str]):
        self.supported_formats = supported_formats
        self.processed_files = {}
        self.metrics_collector = MetricsCollector()

    def scan_directory(self, directory: str) -> List[str]:
        """Scan directory for supported video files."""
        video_files = []
        try:
            for root, _, files in os.walk(directory):
                for file in sorted(files):
                    if any(file.lower().endswith(fmt) for fmt in self.supported_formats):
                        full_path = os.path.join(root, file)
                        video_files.append(full_path)
                        self._collect_file_metrics(full_path)
            return video_files
        except Exception as e:
            log_event(f"Error scanning directory {directory}: {e}", "ERROR", "FILE_SCAN")
            return []

    def _collect_file_metrics(self, file_path: str) -> None:
        """Collect metrics for a video file."""
        try:
            size = os.path.getsize(file_path)
            clip = mp.VideoFileClip(file_path)
            duration = clip.duration
            frames = int(duration * clip.fps)
            
            metrics = {
                'size': size,
                'duration': duration,
                'frames': frames,
                'fps': clip.fps,
                'resolution': (clip.w, clip.h)
            }
            
            self.metrics_collector.update_file_metrics(file_path, metrics)
            self.processed_files[file_path] = metrics
            
            log_event(
                f"File metrics - {os.path.basename(file_path)}: "
                f"Duration={duration:.2f}s, Size={size/(1024*1024):.2f}MB, "
                f"Frames={frames}",
                "INFO",
                "METRICS"
            )
            
            clip.close()
                     
        except Exception as e:
            log_event(f"Error collecting metrics for {file_path}: {e}", "ERROR", "METRICS")

class LogManager:
    """Enhanced logging system with categories and levels."""
    def __init__(self, log_file: str):
        self.log_file = log_file
        self.log_levels = {
            'DEBUG': 0,
            'INFO': 1,
            'WARNING': 2,
            'ERROR': 3
        }
        self.current_level = 'INFO'
        self.lock = Lock()
        self._ensure_log_file()

    def _ensure_log_file(self) -> None:
        """Ensure log file exists and is ready."""
        try:
            if not os.path.exists(os.path.dirname(self.log_file)):
                os.makedirs(os.path.dirname(self.log_file))
            if not os.path.exists(self.log_file):
                with open(self.log_file, 'w') as f:
                    f.write(f"Log initialized: {datetime.datetime.now()}\n")
        except Exception as e:
            print(f"Error initializing log file: {e}")

    def log(self, message: str, level: str = 'INFO', category: str = 'GENERAL') -> None:
        """Log a message with level and category."""
        if self.log_levels.get(level, 0) >= self.log_levels.get(self.current_level, 0):
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_entry = f"[{timestamp}] [{level}] [{category}] {message}"
            
            with self.lock:
                try:
                    with open(self.log_file, 'a') as f:
                        f.write(log_entry + '\n')
                except Exception as e:
                    print(f"Error writing to log: {e}")
                    print(log_entry)

    def get_recent_logs(self, num_lines: int = 50, level: Optional[str] = None,
                       category: Optional[str] = None) -> List[str]:
        """Get recent log entries with optional filtering."""
        with self.lock:
            try:
                with open(self.log_file, 'r') as f:
                    lines = f.readlines()
                
                filtered_lines = []
                for line in reversed(lines):
                    if level and f"[{level}]" not in line:
                        continue
                    if category and f"[{category}]" not in line:
                        continue
                    filtered_lines.append(line.strip())
                    if len(filtered_lines) >= num_lines:
                        break
                        
                return list(reversed(filtered_lines))
            except Exception as e:
                return [f"Error reading logs: {e}"]

    def clear_logs(self) -> None:
        """Clear the log file."""
        with self.lock:
            try:
                with open(self.log_file, 'w') as f:
                    f.write(f"Log cleared: {datetime.datetime.now()}\n")
            except Exception as e:
                print(f"Error clearing logs: {e}")

class ProgressTracker:
    """Track processing progress and status."""
    def __init__(self, total_steps: int):
        self.total_steps = total_steps
        self.current_step = 0
        self.start_time = time.time()
        self.phase = "Initializing"
        self.metrics_collector = MetricsCollector()
        
    def update(self, step: int = 1, phase: Optional[str] = None) -> None:
        """Update progress and optionally change phase."""
        self.current_step += step
        if phase:
            self.phase = phase
            self.metrics_collector.start_phase_timing(phase)
        self._log_progress()
        
    def _log_progress(self) -> None:
        """Log progress update with memory usage."""
        progress = (self.current_step / self.total_steps) * 100
        elapsed = time.time() - self.start_time
        memory = psutil.Process().memory_info().rss
        
        self.metrics_collector.update_processing_metrics(1, memory)
        log_event(
            f"Progress: {progress:.1f}% - Phase: {self.phase} - "
            f"Elapsed: {elapsed:.1f}s - Memory: {memory/(1024*1024):.1f}MB",
            "INFO",
            "PROGRESS"
        )
        
    def get_progress(self) -> Dict[str, Any]:
        """Get current progress information."""
        elapsed = time.time() - self.start_time
        progress = (self.current_step / self.total_steps) * 100
        
        return {
            "progress": progress,
            "phase": self.phase,
            "elapsed": f"{int(elapsed//3600):02d}:{int((elapsed%3600)//60):02d}:"
                      f"{int(elapsed%60):02d}",
            "metrics": self.metrics_collector.get_metrics_report()
        }

def monitor_memory_usage(threshold_mb: float = 1000.0) -> bool:
    """Monitor memory usage and log warnings if threshold exceeded."""
    try:
        current_memory = psutil.Process().memory_info().rss / (1024 * 1024)  # Convert to MB
        if current_memory > threshold_mb:
            log_event(
                f"High memory usage detected: {current_memory:.1f}MB",
                "WARNING",
                "MEMORY"
            )
            return True
        return False
    except Exception as e:
        log_event(f"Error monitoring memory: {e}", "ERROR", "MEMORY")
        return False

def cleanup_work_directory() -> None:
    """Clean up temporary files in work directory."""
    try:
        work_dir = os.path.join("work")
        if os.path.exists(work_dir):
            shutil.rmtree(work_dir)
            os.makedirs(work_dir)
            log_event("Work directory cleaned", "INFO", "CLEANUP")
    except Exception as e:
        log_event(f"Error cleaning work directory: {e}", "ERROR", "CLEANUP")

def load_hardware_config() -> Dict[str, bool]:
    """Load hardware configuration from hardware.txt."""
    hardware_file = os.path.join("data", "hardware.txt")
    hardware_config = {
        "x64": False,
        "Avx2": False,
        "Aocl": False,
        "OpenCL": False,
    }
    try:
        with open(hardware_file, "r") as f:
            for line in f:
                key, value = line.strip().split(": ")
                hardware_config[key] = value.lower() == "true"
        log_event("Hardware configuration loaded", "INFO", "CONFIG")
    except Exception as e:
        log_event(f"Error loading hardware config: {e}", "ERROR", "CONFIG")
    return hardware_config

def load_settings() -> Dict[str, Any]:
    """Load settings from temporary.py."""
    try:
        from data.temporary import SEARCH_CRITERIA, HARDWARE_CONFIG, VIDEO_CONFIG
        log_event("Settings loaded from temporary.py", "INFO", "CONFIG")
        return {
            "search": SEARCH_CRITERIA,
            "hardware": HARDWARE_CONFIG,
            "video": VIDEO_CONFIG
        }
    except Exception as e:
        log_event(f"Error loading settings: {e}", "ERROR", "CONFIG")
        return {}

def log_event(message: str, level: str = 'INFO', category: str = 'GENERAL') -> None:
    """Log an event to the event log file."""
    log_file = os.path.join("data", "events.txt")
    try:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] [{level}] [{category}] {message}\n"
        
        with open(log_file, 'a') as f:
            f.write(log_entry)
    except Exception as e:
        print(f"Error logging event: {e}")
        print(log_entry)
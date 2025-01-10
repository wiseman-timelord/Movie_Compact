# .\scripts\utility.py

import os, cv2, numpy as np, pyopencl as cl, json, datetime, sys, psutil
from typing import Tuple, List, Dict, Any, Generator, Optional
import moviepy.editor as mp
from interface import HardwareError, ConfigurationError, MovieConsolidatorError

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
        report.append(f"Total Duration Processed: {self.metrics['total_duration']:.2f} seconds")
        report.append(f"Total Size Processed: {self.metrics['total_size'] / (1024*1024):.2f} MB")
        report.append(f"Total Frames Processed: {self.metrics['processed_frames']}")
        report.append(f"Peak Memory Usage: {self.metrics['memory_usage'] / (1024*1024):.2f} MB")
        
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
            report.append(f"    Size: {metrics.get('size', 0) / (1024*1024):.2f} MB")
            report.append(f"    Frames: {metrics.get('frames', 0)}")
        
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
            clip.close()

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
                f"Duration={duration:.2f}s, Size={size/(1024*1024):.2f}MB, Frames={frames}",
                "INFO", 
                "METRICS"
            )
                     
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
        self._ensure_log_file()

    def _ensure_log_file(self) -> None:
        """Ensure log file exists and is ready."""
        if not os.path.exists(os.path.dirname(self.log_file)):
            os.makedirs(os.path.dirname(self.log_file))
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w') as f:
                f.write(f"Log initialized: {datetime.datetime.now()}\n")

    def log(self, message: str, level: str = 'INFO', category: str = 'GENERAL') -> None:
        """Log a message with level and category."""
        if self.log_levels.get(level, 0) >= self.log_levels.get(self.current_level, 0):
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_entry = f"[{timestamp}] [{level}] [{category}] {message}"
            
            try:
                with open(self.log_file, 'a') as f:
                    f.write(log_entry + '\n')
            except Exception as e:
                print(f"Error writing to log: {e}")
                print(log_entry)

    def get_recent_logs(self, num_lines: int = 20, level: str = None, 
                       category: str = None) -> List[str]:
        """Get recent log entries with optional filtering."""
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
        try:
            with open(self.log_file, 'w') as f:
                f.write(f"Log cleared: {datetime.datetime.now()}\n")
        except Exception as e:
            print(f"Error clearing logs: {e}")

class ProgressTracker:
    def __init__(self, total_steps: int):
        self.total_steps = total_steps
        self.current_step = 0
        self.start_time = datetime.datetime.now()
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
        elapsed = datetime.datetime.now() - self.start_time
        memory = psutil.Process().memory_info().rss
        self.metrics_collector.update_processing_metrics(1, memory)
        log_event(
            f"Progress: {progress:.1f}% - Phase: {self.phase} - "
            f"Elapsed: {elapsed} - Memory: {memory/(1024*1024):.1f}MB",
            "INFO",
            "PROGRESS"
        )
        
    def get_progress(self) -> Dict[str, Any]:
        """Get current progress information."""
        return {
            "progress": (self.current_step / self.total_steps) * 100,
            "phase": self.phase,
            "elapsed": str(datetime.datetime.now() - self.start_time),
            "metrics": self.metrics_collector.get_metrics_report()
        }

def monitor_memory_usage(threshold_mb: float = 1000.0) -> None:
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

# Initialize the log manager
log_manager = LogManager(os.path.join("data", "events.txt"))

def log_event(message: str, level: str = 'INFO', category: str = 'GENERAL') -> None:
    """Enhanced logging function using LogManager."""
    log_manager.log(message, level, category)

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
        log_event("Hardware configuration loaded successfully", "INFO", "CONFIG")
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

def initialize_opencl_with_fallback() -> Tuple[Any, Any, str]:
    """Initialize OpenCL with robust fallback mechanism."""
    try:
        platforms = cl.get_platforms()
        settings = load_settings()
        preferred_platforms = settings.get('hardware', {}).get('opencl_platform_preference', ['NVIDIA', 'AMD', 'Intel'])
        
        # Try preferred platforms in order
        for pref in preferred_platforms:
            for platform in platforms:
                if pref.lower() in platform.name.lower():
                    try:
                        devices = platform.get_devices()
                        context = cl.Context(devices)
                        queue = cl.CommandQueue(context)
                        log_event(f"OpenCL initialized using {platform.name}", "INFO", "HARDWARE")
                        return context, queue, platform.name
                    except Exception:
                        continue
        
        # Try any available platform
        if platforms:
            try:
                platform = platforms[0]
                devices = platform.get_devices()
                context = cl.Context(devices)
                queue = cl.CommandQueue(context)
                log_event(f"OpenCL initialized using fallback platform: {platform.name}", "INFO", "HARDWARE")
                return context, queue, platform.name
            except Exception:
                pass
        
        log_event("OpenCL initialization failed, falling back to CPU", "WARNING", "HARDWARE")
        return None, None, "CPU"
        
    except Exception as e:
        log_event(f"OpenCL initialization failed: {e}", "ERROR", "HARDWARE")
        return None, None, "CPU"

def get_video_files(directory: str) -> List[str]:
    """Get list of video files from directory."""
    settings = load_settings()
    supported_formats = settings.get('video', {}).get('supported_formats', ['.mp4', '.avi', '.mkv'])
    
    try:
        video_files = [
            os.path.join(directory, f) for f in os.listdir(directory)
            if os.path.splitext(f)[1].lower() in supported_formats
        ]
        log_event(f"Found {len(video_files)} video files in {directory}", "INFO", "FILES")
        return sorted(video_files)
    except Exception as e:
        log_event(f"Error listing video files: {e}", "ERROR", "FILES")
        return []

def get_video_duration(file_path: str) -> float:
    """Get duration of video file."""
    try:
        clip = mp.VideoFileClip(file_path)
        duration = clip.duration
        clip.close()
        return duration
    except Exception as e:
        log_event(f"Error getting video duration for {file_path}: {e}", "ERROR", "FILES")
        return 0.0

def extract_frames_optimized(video_path: str, frame_interval: int = 1) -> Generator[np.ndarray, None, None]:
    """Memory-optimized frame extraction using generator."""
    try:
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        memory_usage = psutil.Process().memory_info().rss
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % frame_interval == 0:
                yield frame
            frame_count += 1
            
            # Monitor memory usage
                current_memory = psutil.Process().memory_info().rss
                if current_memory > memory_usage * 1.5:
                    log_event(f"High memory usage detected: {current_memory/(1024*1024):.1f}MB", "WARNING", "MEMORY")
            
        cap.release()
        log_event(f"Extracted frames from {video_path} with interval {frame_interval}", "INFO", "PROCESSING")
        
    except Exception as e:
        log_event(f"Frame extraction failed for {video_path}: {e}", "ERROR", "PROCESSING")
        yield None

def batch_process_frames(frames: List[np.ndarray], batch_size: int, process_func: Any, **kwargs) -> List[Any]:
    """Process frames in batches to optimize memory usage."""
    results = []
    memory_start = psutil.Process().memory_info().rss
    
    for i in range(0, len(frames), batch_size):
        batch = frames[i:i+batch_size]
        batch_results = [process_func(frame, **kwargs) for frame in batch]
        results.extend(batch_results)
        
        # Monitor memory
        current_memory = psutil.Process().memory_info().rss
        if current_memory > memory_start * 1.5:
            log_event(f"High memory usage in batch processing: {current_memory/(1024*1024):.1f}MB", "WARNING", "MEMORY")
            
    return results

def detect_motion_opencl(frame1: np.ndarray, frame2: np.ndarray, threshold: float) -> bool:
    """Detect motion between frames using OpenCL."""
    try:
        context, queue, platform_name = initialize_opencl_with_fallback()
        if not context:
            return detect_motion_cpu(frame1, frame2, threshold)
            
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        mf = cl.mem_flags
        gray1_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=gray1)
        gray2_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=gray2)
        diff_buf = cl.Buffer(context, mf.WRITE_ONLY, gray1.nbytes)
        
        kernel_code = """
        __kernel void detect_motion(
            __global const uchar* gray1,
            __global const uchar* gray2,
            __global uchar* diff
        ) {
            int gid = get_global_id(0);
            diff[gid] = (uchar)abs((int)gray1[gid] - (int)gray2[gid]);
        }
        """
        
        program = cl.Program(context, kernel_code).build()
        program.detect_motion(queue, gray1.shape, None, gray1_buf, gray2_buf, diff_buf)
        
        diff = np.empty_like(gray1)
        cl.enqueue_copy(queue, diff, diff_buf)
        motion_score = np.mean(diff) / 255.0
        
        return motion_score > threshold
        
    except Exception as e:
        log_event(f"OpenCL motion detection failed: {e}", "ERROR", "PROCESSING")
        return detect_motion_cpu(frame1, frame2, threshold)

def detect_motion_cpu(frame1: np.ndarray, frame2: np.ndarray, threshold: float) -> bool:
    """Detect motion between frames using CPU."""
    try:
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(gray1, gray2)
        _, diff = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        motion_score = np.mean(diff) / 255.0
        return motion_score > threshold
    except Exception as e:
        log_event(f"CPU motion detection failed: {e}", "ERROR", "PROCESSING")
        return False

def detect_texture_change(frame1: np.ndarray, frame2: np.ndarray, threshold: float) -> bool:
    """Detect texture changes between frames."""
    try:
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        hist1 = cv2.calcHist([gray1], [0], None, [256], [0, 256])
        hist2 = cv2.calcHist([gray2], [0], None, [256], [0, 256])
        
        texture_score = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CHISQR)
        return texture_score > threshold
    except Exception as e:
        log_event(f"Texture detection failed: {e}", "ERROR", "PROCESSING")
        return False

def analyze_segment(frame1: np.ndarray, frame2: np.ndarray) -> bool:
    """Analyze segment for motion or texture changes."""
    try:
        settings = load_settings()
        search_criteria = settings.get('search', {})
        
        motion_threshold = search_criteria.get('motion_threshold', 0.5)
        texture_threshold = search_criteria.get('texture_threshold', 0.6)
        
        hardware_config = load_hardware_config()
        if hardware_config["OpenCL"]:
            motion = detect_motion_opencl(frame1, frame2, motion_threshold)
        else:
            motion = detect_motion_cpu(frame1, frame2, motion_threshold)
            
        texture = detect_texture_change(frame1, frame2, texture_threshold)
        return motion or texture
        
    except Exception as e:
        log_event(f"Segment analysis failed: {e}", "ERROR", "PROCESSING")
        return False

def extract_frames(video_path: str) -> List[np.ndarray]:
    """Extract frames from video at specified frame rate."""
    try:
        settings = load_settings()
        frame_rate = settings.get('search', {}).get('frame_settings', {}).get('sample_rate', 30)
        
        frames = list(extract_frames_optimized(video_path, frame_rate))
        log_event(f"Extracted {len(frames)} frames from {video_path}", "INFO", "PROCESSING")
        return frames
        
    except Exception as e:
        log_event(f"Frame extraction failed for {video_path}: {e}", "ERROR", "PROCESSING")
        return []

def cleanup_work_directory() -> None:
    """Clean up temporary files in work directory."""
    try:
        work_dir = "work"
        for filename in os.listdir(work_dir):
            file_path = os.path.join(work_dir, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
        log_event("Work directory cleaned", "INFO", "CLEANUP")
    except Exception as e:
        log_event(f"Work directory cleanup failed: {e}", "ERROR", "CLEANUP")
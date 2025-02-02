# .\scripts\analyze.py

import cv2
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Generator, Union
from dataclasses import dataclass
from utility import (
    analyze_segment, extract_frames, load_settings, log_event,
    extract_frames_optimized, detect_motion_opencl, detect_motion_cpu,
    detect_texture_change, SceneManager, AudioAnalyzer, PreviewGenerator,
    monitor_memory_usage, MemoryManager, ProgressMonitor, ErrorHandler,
    CoreUtilities
)
from interface import AnalysisError, MovieCompactError
import time
from collections import deque
from scipy import signal
import librosa

@dataclass
class AnalysisConfig:
    """Configuration for video analysis."""
    motion_threshold: float = 0.3
    texture_threshold: float = 0.4
    static_threshold: float = 0.95
    menu_threshold: float = 0.7
    action_threshold: float = 0.6
    min_scene_duration: float = 2.0

class ContentAnalyzer:
    """Advanced content analysis for video frames."""
    
    def __init__(self):
        self.settings = load_settings()
        self.frame_cache = {}
    
    def analyze_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """Single entry point for frame analysis."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        return {
            'text_regions': self._detect_text(gray),
            'ui_elements': self._detect_ui(edges, gray),
            'motion': self._analyze_motion(gray),
            'complexity': self._calculate_complexity(gray, edges),
            'color_profile': self._analyze_colors(hsv),
            'frame_type': self._determine_type(gray, edges, hsv)
        }
    
    def _detect_text(self, gray: np.ndarray) -> List[Dict[str, Any]]:
        """Detect text regions in frame."""
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, 11, 2)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, 
                                     cv2.CHAIN_APPROX_SIMPLE)
        return [{'bbox': cv2.boundingRect(c)} for c in contours 
                if 0.1 < cv2.boundingRect(c)[2] / cv2.boundingRect(c)[3] < 15]
    
    def _detect_ui(self, edges: np.ndarray, gray: np.ndarray) -> List[Dict[str, Any]]:
        """Detect UI elements."""
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, 
                                     cv2.CHAIN_APPROX_SIMPLE)
        ui_elements = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if 0.8 < w/h < 1.2:
                ui_elements.append({
                    'bbox': (x, y, w, h),
                    'type': 'button' if w/h < 1.1 else 'panel'
                })
        return ui_elements
    
    def _analyze_motion(self, gray: np.ndarray) -> Dict[str, float]:
        """Analyze motion in frame."""
        if not hasattr(self, 'prev_gray'):
            self.prev_gray = gray
            return {'magnitude': 0, 'direction': 0}
        
        flow = cv2.calcOpticalFlowFarneback(self.prev_gray, gray, None, 
                                          0.5, 3, 15, 3, 5, 1.2, 0)
        self.prev_gray = gray
        magnitude = np.mean(np.sqrt(flow[..., 0]**2 + flow[..., 1]**2))
        direction = np.mean(np.arctan2(flow[..., 1], flow[..., 0]))
        return {'magnitude': float(magnitude), 'direction': float(direction)}
    
    def _calculate_complexity(self, gray: np.ndarray, edges: np.ndarray) -> float:
        """Calculate frame complexity."""
        edge_density = np.count_nonzero(edges) / edges.size
        texture = cv2.Laplacian(gray, cv2.CV_64F).var()
        return float(0.5 * edge_density + 0.5 * min(texture/1000, 1.0))
    
    def _analyze_colors(self, hsv: np.ndarray) -> Dict[str, Any]:
        """Analyze color distribution."""
        hist_h = cv2.calcHist([hsv], [0], None, [180], [0, 180])
        hist_s = cv2.calcHist([hsv], [1], None, [256], [0, 256])
        return {
            'hue_dist': hist_h.flatten().tolist(),
            'sat_dist': hist_s.flatten().tolist(),
            'brightness': float(np.mean(hsv[..., 2]))
        }
    
    def _determine_type(self, gray: np.ndarray, edges: np.ndarray, 
                       hsv: np.ndarray) -> str:
        """Determine frame type."""
        edge_density = np.count_nonzero(edges) / edges.size
        brightness_var = np.std(hsv[..., 2])
        text_like = len(self._detect_text(gray))
        
        if edge_density < 0.1 and brightness_var < 30:
            return 'static'
        elif text_like > 10 and edge_density < 0.3:
            return 'menu'
        elif edge_density > 0.4 and brightness_var > 50:
            return 'action'
        return 'gameplay'

class VideoAnalyzer:
    """Handles video analysis operations."""
    
    def __init__(self, log_manager=None):
        self.core = CoreUtilities()
        self.config = AnalysisConfig(**self.core.settings.get('analysis', {}))
        self.content_analyzer = ContentAnalyzer()
        self.scene_manager = SceneManager()
        self.audio_analyzer = AudioAnalyzer()
        self.preview_generator = PreviewGenerator()
        self.memory_manager = MemoryManager()
        self.progress = ProgressMonitor()
        self.error_handler = ErrorHandler()
        self.log_manager = log_manager
        self.frame_buffer = deque(maxlen=30)
    
    def analyze_video(self, video_path: str, target_duration: float) -> Dict[str, Any]:
        """Analyze video content and structure."""
        try:
            self.progress.start_stage("Analysis")
            
            # Initial setup and preview generation
            preview_path = self._setup_analysis(video_path)
            if not preview_path:
                raise AnalysisError("Failed to create preview")
            
            # Extract and analyze content
            frames, audio_data = self._extract_content(preview_path)
            if not frames:
                raise AnalysisError("Failed to extract frames")
            
            # Process scenes
            scenes = self._process_scenes(frames, audio_data, target_duration)
            
            # Calculate final statistics
            stats = self._calculate_statistics(scenes)
            
            self.progress.complete_stage("Analysis")
            self.memory_manager.cleanup()
            
            return {
                'scenes': scenes,
                'stats': stats,
                'frame_rate': self._get_frame_rate(preview_path),
                'total_frames': len(frames)
            }
            
        except Exception as e:
            self.error_handler.handle_error(e, "video_analysis")
            return {'scenes': [], 'stats': {}, 'frame_rate': 0, 'total_frames': 0}
    
    def _setup_analysis(self, video_path: str) -> Optional[str]:
        """Setup analysis environment."""
        self.progress.update_progress(0, "Creating preview")
        return self.preview_generator.create_preview(video_path)
    
    def _extract_content(self, video_path: str) -> Tuple[List[np.ndarray], np.ndarray]:
        """Extract frames and audio."""
        self.progress.update_progress(20, "Extracting content")
        frames = list(extract_frames_optimized(video_path))
        audio = self.audio_analyzer.extract_audio(video_path)
        return frames, audio
    
    def _process_scenes(self, frames: List[np.ndarray], audio: np.ndarray,
                       target_duration: float) -> List[Dict[str, Any]]:
        """Process and analyze scenes."""
        self.progress.update_progress(40, "Processing scenes")
        scenes = []
        
        for i, frame_group in enumerate(self._group_frames(frames)):
            if self.memory_manager.check_memory()['warning']:
                self.memory_manager.cleanup()
            
            scene = self._analyze_scene(frame_group, audio, target_duration, i/len(frames))
            scenes.append(scene)
            
            progress = 40 + (i + 1) / len(frames) * 40
            self.progress.update_progress(progress, f"Processed scene {i+1}")
        
        return self._merge_scenes(scenes)
    
    def _group_frames(self, frames: List[np.ndarray]) -> Generator[List[np.ndarray], None, None]:
        """Group frames into potential scenes."""
        scene_start = 0
        for i in range(1, len(frames)):
            if detect_motion_opencl(frames[i-1], frames[i], self.config.motion_threshold):
                if i - scene_start >= self.config.min_scene_duration * 30:  # Assuming 30fps
                    yield frames[scene_start:i]
                    scene_start = i
        if scene_start < len(frames):
            yield frames[scene_start:]
    
    def _analyze_scene(self, frames: List[np.ndarray], audio: np.ndarray,
                      target_duration: float, position: float) -> Dict[str, Any]:
        """Analyze a single scene."""
        analysis = self.content_analyzer.analyze_frame(frames[len(frames)//2])  # Middle frame
        
        return {
            'start_frame': frames[0],
            'end_frame': frames[-1],
            'type': analysis['frame_type'],
            'complexity': analysis['complexity'],
            'motion': analysis['motion'],
            'speed_factor': self._calculate_speed(analysis, target_duration, position)
        }
    
    def _calculate_speed(self, analysis: Dict[str, Any], target_duration: float,
                        position: float) -> float:
        """Calculate scene speed factor."""
        if analysis['frame_type'] == 'action':
            return 1.0
        elif analysis['frame_type'] == 'menu':
            return min(4.0, 2.0)
        
        base_speed = 1.0 + position * 0.5
        content_factor = 1.0 - min(analysis['complexity'], 0.8)
        return min(4.0, max(1.0, base_speed * (1.0 + content_factor)))
    
    def _merge_scenes(self, scenes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge similar adjacent scenes."""
        merged = []
        current = scenes[0]
        
        for next_scene in scenes[1:]:
            if (next_scene['type'] == current['type'] and
                abs(next_scene['complexity'] - current['complexity']) < 0.2):
                current['end_frame'] = next_scene['end_frame']
                current['complexity'] = (current['complexity'] + next_scene['complexity']) / 2
            else:
                merged.append(current)
                current = next_scene
        
        merged.append(current)
        return merged
    
    def _calculate_statistics(self, scenes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate final statistics."""
        return {
            'total_scenes': len(scenes),
            'scene_types': {type_: len([s for s in scenes if s['type'] == type_])
                          for type_ in set(s['type'] for s in scenes)},
            'average_complexity': sum(s['complexity'] for s in scenes) / len(scenes),
            'speed_distribution': {
                'normal': len([s for s in scenes if s['speed_factor'] == 1.0]),
                'fast': len([s for s in scenes if s['speed_factor'] > 1.0])
            }
        }
    
    @staticmethod
    def _get_frame_rate(video_path: str) -> float:
        """Get video frame rate."""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        return fps

def analyze_video_file(path: str, target_duration: float) -> Dict[str, Any]:
    """Analyze a single video file."""
    return VideoAnalyzer().analyze_video(path, target_duration)

def estimate_processing_time(path: str) -> float:
    """Estimate processing time."""
    cap = cv2.VideoCapture(path)
    duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
    _, frame = cap.read()
    cap.release()
    
    base_time = duration * 0.1
    if frame is not None:
        resolution_factor = (frame.shape[0] * frame.shape[1]) / (1280 * 720)
        base_time *= min(resolution_factor, 2.0)
    
    return base_time

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        path = sys.argv[1]
        print(f"Analyzing: {path}")
        target_duration = float(sys.argv[2]) if len(sys.argv) > 2 else None
        results = analyze_video_file(path, target_duration)
        print(f"Found {len(results['scenes'])} scenes")
        print(f"Analysis complete: {results['stats']}")
    else:
        print("Usage: python analyze.py <video_file> [target_duration]")
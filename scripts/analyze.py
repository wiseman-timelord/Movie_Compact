# .\scripts\analyze.py

import cv2
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from utility import (
    analyze_segment,
    extract_frames,
    load_settings,
    log_event,
    extract_frames_optimized,
    detect_motion_opencl,
    detect_motion_cpu,
    detect_texture_change
)
from interface import AnalysisError, MovieConsolidatorError

class VideoAnalyzer:
    """Handles video analysis operations including scene detection and speed calculations."""
    def __init__(self):
        self.settings = load_settings()
        self.search_criteria = self.settings.get('search', {})
        self.video_config = self.settings.get('video', {})

    def detect_static_frame(self, frame1: np.ndarray, frame2: np.ndarray, 
                          threshold: float = 0.99) -> bool:
        """Detect if frames are static (nearly identical)."""
        try:
            diff = cv2.absdiff(frame1, frame2)
            diff_score = 1 - (np.count_nonzero(diff) / diff.size)
            return diff_score > threshold
        except Exception as e:
            log_event(f"Error in static frame detection: {e}", "ERROR", "DETECTION")
            return False

    def detect_menu_screen(self, frame: np.ndarray) -> bool:
        """Detect if frame is likely a menu screen."""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.count_nonzero(edges) / edges.size
            return edge_density > 0.1
        except Exception as e:
            log_event(f"Error in menu detection: {e}", "ERROR", "DETECTION")
            return False

    def detect_low_activity(self, frames: List[np.ndarray], 
                          threshold: float = 0.1) -> bool:
        """Detect sections with low activity."""
        try:
            motion_scores = []
            for i in range(1, len(frames)):
                diff = cv2.absdiff(frames[i-1], frames[i])
                motion_score = np.mean(diff) / 255.0
                motion_scores.append(motion_score)
            
            average_motion = np.mean(motion_scores)
            log_event(f"Low activity detection: score={average_motion:.3f}", 
                     "DEBUG", "DETECTION")
            return average_motion < threshold
            
        except Exception as e:
            log_event(f"Error in low activity detection: {e}", "ERROR", "DETECTION")
            return False

    def detect_scene_change(self, frame1: np.ndarray, frame2: np.ndarray) -> bool:
        """Detect scene changes between frames."""
        try:
            threshold = self.settings.get('scene_settings', {}).get('scene_threshold', 30.0)
            hist1 = cv2.calcHist([frame1], [0, 1, 2], None, [8, 8, 8], 
                                [0, 256, 0, 256, 0, 256])
            hist2 = cv2.calcHist([frame2], [0, 1, 2], None, [8, 8, 8], 
                                [0, 256, 0, 256, 0, 256])
            diff = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CHISQR)
            
            if diff > threshold:
                log_event(f"Scene change detected: score={diff:.2f}", "DEBUG", "DETECTION")
            
            return diff > threshold
            
        except Exception as e:
            log_event(f"Error in scene change detection: {e}", "ERROR", "DETECTION")
            return False

    def detect_action_sequence(self, frames: List[np.ndarray]) -> bool:
        """Detect action sequences based on motion and color changes."""
        try:
            threshold = self.settings.get('scene_settings', {}).get('action_threshold', 0.3)
            motion_scores = []
            
            for i in range(1, len(frames)):
                diff = cv2.absdiff(frames[i-1], frames[i])
                motion_score = np.mean(diff) / 255.0
                motion_scores.append(motion_score)
            
            average_motion = np.mean(motion_scores)
            if average_motion > threshold:
                log_event(f"Action sequence detected: score={average_motion:.3f}", 
                         "INFO", "DETECTION")
            
            return average_motion > threshold
            
        except Exception as e:
            log_event(f"Error in action sequence detection: {e}", "ERROR", "DETECTION")
            return False

    def calculate_speed_factor(self, scene_duration: float, total_duration: float, 
                             target_duration: float, scene_position: float,
                             is_action: bool) -> float:
        """Calculate appropriate speed factor for a scene."""
        try:
            speed_settings = self.settings.get('speed_settings', {})
            max_speed = speed_settings.get('max_speed_factor', 4.0)
            min_speed = speed_settings.get('min_speed_factor', 1.0)
            
            if is_action:
                return 1.0  # Keep action scenes at normal speed
            
            # Calculate required overall speed factor
            overall_speed_needed = total_duration / target_duration
            
            # Calculate dynamic speed factor based on scene position
            # Scenes in the middle get higher speed factors
            position_factor = 1.0 + (scene_position - 0.5) ** 2
            
            # Calculate base speed factor
            base_factor = overall_speed_needed * position_factor
            
            # Adjust for scene duration - longer scenes can be sped up more
            duration_factor = min(scene_duration / 10.0, 1.0)  # Scale based on duration
            
            # Clamp between min and max speed
            factor = max(min_speed, min(max_speed, base_factor))
            
            log_event(
                f"Calculated speed factor: {factor:.2f}x for {scene_duration:.2f}s scene",
                "DEBUG",
                "SPEED"
            )
            
            return factor
            
        except Exception as e:
            log_event(f"Error calculating speed factor: {e}", "ERROR", "SPEED")
            return 1.0

    def analyze_video(self, frames: List[np.ndarray], target_duration: float) -> Dict[str, Any]:
        """Perform complete analysis of video frames."""
        try:
            scenes = []
            current_scene_start = 0
            
            # Detect scenes and their characteristics
            for i in range(1, len(frames)):
                if self.detect_scene_change(frames[i-1], frames[i]):
                    scene_frames = frames[current_scene_start:i]
                    is_action = self.detect_action_sequence(scene_frames)
                    
                    scenes.append({
                        'start_frame': current_scene_start,
                        'end_frame': i,
                        'is_action': is_action,
                        'is_menu': self.detect_menu_screen(frames[i-1]),
                        'is_static': self.detect_static_frame(
                            frames[max(0, i-2)], frames[i-1]
                        ),
                        'is_low_activity': self.detect_low_activity(scene_frames)
                    })
                    current_scene_start = i
            
            # Add final scene if exists
            if current_scene_start < len(frames) - 1:
                final_frames = frames[current_scene_start:]
                scenes.append({
                    'start_frame': current_scene_start,
                    'end_frame': len(frames) - 1,
                    'is_action': self.detect_action_sequence(final_frames),
                    'is_menu': self.detect_menu_screen(frames[-1]),
                    'is_static': self.detect_static_frame(
                        frames[-2], frames[-1]
                    ),
                    'is_low_activity': self.detect_low_activity(final_frames)
                })
            
            log_event(f"Completed analysis of {len(scenes)} scenes", "INFO", "ANALYSIS")
            return {'scenes': scenes}
            
        except Exception as e:
            log_event(f"Error in video analysis: {e}", "ERROR", "ANALYSIS")
            return {'scenes': []}

# Create analyzer instance
analyzer = VideoAnalyzer()

def analyze_video_file(video_path: str, target_duration: float) -> Dict[str, Any]:
    """Analyze a single video file."""
    try:
        frames = list(extract_frames_optimized(video_path))
        return analyzer.analyze_video(frames, target_duration)
    except Exception as e:
        log_event(f"Error analyzing video file: {e}", "ERROR", "ANALYSIS")
        return {'scenes': []}
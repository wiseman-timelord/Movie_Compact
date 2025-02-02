# analyze.py

import cv2
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Generator
from utility import (
    analyze_segment,
    extract_frames,
    load_settings,
    log_event,
    extract_frames_optimized,
    detect_motion_opencl,
    detect_motion_cpu,
    detect_texture_change,
    SceneManager,
    AudioAnalyzer,
    PreviewGenerator,
    monitor_memory_usage
)
from interface import AnalysisError, MovieCompactError
import time
from collections import deque
from scipy import signal

class VideoAnalyzer:
    """Handles video analysis operations including scene detection and speed calculations."""
    
    def __init__(self, log_manager=None):
        self.settings = load_settings()
        self.search_criteria = self.settings.get('search', {})
        self.video_config = self.settings.get('video', {})
        self.scene_manager = SceneManager()
        self.audio_analyzer = AudioAnalyzer()
        self.preview_generator = PreviewGenerator()
        self.log_manager = log_manager
        
        # Analysis parameters
        self.motion_threshold = self.search_criteria.get('motion_threshold', 0.3)
        self.texture_threshold = self.search_criteria.get('texture_threshold', 0.4)
        self.static_threshold = self.search_criteria.get('static_threshold', 0.95)
        self.menu_threshold = self.search_criteria.get('menu_threshold', 0.7)
        self.action_threshold = self.search_criteria.get('action_threshold', 0.6)
        self.min_scene_duration = self.search_criteria.get('min_scene_duration', 2.0)
        
        # Initialize frame buffers for analysis
        self.frame_buffer = deque(maxlen=30)  # For motion analysis
        self.scene_buffer = deque(maxlen=5)   # For scene transition detection

    def log(self, message: str, level: str = "INFO", category: str = "ANALYSIS"):
        """Log a message using the log manager if available."""
        if self.log_manager:
            self.log_manager.log(message, level, category)
        else:
            log_event(message, level, category)

    def detect_static_frame(self, frame1: np.ndarray, frame2: np.ndarray) -> bool:
        """Detect if frames are static (nearly identical)."""
        try:
            # Convert to grayscale for faster comparison
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            
            # Calculate absolute difference
            diff = cv2.absdiff(gray1, gray2)
            diff_score = 1 - (np.count_nonzero(diff) / diff.size)
            
            # Apply additional blur detection
            blur1 = cv2.Laplacian(gray1, cv2.CV_64F).var()
            blur2 = cv2.Laplacian(gray2, cv2.CV_64F).var()
            blur_similar = abs(blur1 - blur2) < 50
            
            is_static = diff_score > self.static_threshold and blur_similar
            
            if is_static:
                self.log(f"Static frame detected, score: {diff_score:.3f}", "DEBUG", "DETECTION")
            
            return is_static
            
        except Exception as e:
            self.log(f"Error in static frame detection: {e}", "ERROR", "DETECTION")
            return False

    def detect_menu_screen(self, frame: np.ndarray) -> bool:
        """Detect if frame is likely a menu screen using advanced detection."""
        try:
            # Convert to grayscale for text detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Edge detection for UI elements
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.count_nonzero(edges) / edges.size
            
            # Text detection using contours
            _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Analyze contours for text-like shapes
            text_like_contours = 0
            rect_like_contours = 0
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(w)/h
                if 0.1 < aspect_ratio < 10:  # Typical text aspect ratio
                    text_like_contours += 1
                if 0.8 < aspect_ratio < 1.2:  # UI element aspect ratio
                    rect_like_contours += 1
            
            text_density = text_like_contours / (frame.shape[0] * frame.shape[1])
            rect_density = rect_like_contours / (frame.shape[0] * frame.shape[1])
            
            # Check for uniform regions (common in menus)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            std_dev = np.std(blur)
            
            # Color analysis for UI elements
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            saturation = np.mean(hsv[:, :, 1])
            value = np.mean(hsv[:, :, 2])
            
            # Combined decision based on multiple factors
            is_menu = (
                edge_density > 0.1 and
                text_density > 0.001 and
                rect_density > 0.0005 and
                std_dev < 50 and        # Low variation in background
                saturation < 100 and    # UI typically has low saturation
                value > 150             # UI typically bright
            )
            
            if is_menu:
                self.log(
                    f"Menu screen detected: edge_density={edge_density:.3f}, "
                    f"text_density={text_density:.3f}",
                    "DEBUG",
                    "DETECTION"
                )
            
            return is_menu
            
        except Exception as e:
            self.log(f"Error in menu detection: {e}", "ERROR", "DETECTION")
            return False

    def detect_low_activity(self, frames: List[np.ndarray]) -> bool:
        """Detect sections with low activity using temporal analysis."""
        try:
            motion_scores = []
            texture_scores = []
            
            for i in range(1, len(frames)):
                # Use OpenCL if available for motion detection
                motion = detect_motion_opencl(frames[i-1], frames[i], self.motion_threshold)
                motion_scores.append(float(motion))
                
                # Check texture changes
                texture = detect_texture_change(frames[i-1], frames[i], self.texture_threshold)
                texture_scores.append(float(texture))
            
            # Calculate weighted average of motion and texture scores
            average_motion = np.mean(motion_scores) if motion_scores else 0
            average_texture = np.mean(texture_scores) if texture_scores else 0
            
            activity_score = 0.7 * average_motion + 0.3 * average_texture
            
            if activity_score < self.motion_threshold:
                self.log(f"Low activity detected: score={activity_score:.3f}",
                        "DEBUG", "DETECTION")
            
            return activity_score < self.motion_threshold
            
        except Exception as e:
            self.log(f"Error in low activity detection: {e}", "ERROR", "DETECTION")
            return False

    def detect_action_sequence(self, frames: List[np.ndarray],
                             audio_segments: List[Tuple[float, float]],
                             frame_rate: float) -> bool:
        """Detect action sequences using multi-modal analysis."""
        try:
            # Visual motion analysis
            motion_scores = []
            color_variation_scores = []
            edge_density_scores = []
            
            for i in range(1, len(frames)):
                # Motion detection
                motion = detect_motion_opencl(frames[i-1], frames[i], self.action_threshold)
                motion_scores.append(float(motion))
                
                # Color variation
                frame_hsv = cv2.cvtColor(frames[i], cv2.COLOR_BGR2HSV)
                color_std = np.std(frame_hsv[:,:,0])  # Hue channel variation
                color_variation_scores.append(color_std)
                
                # Edge density for action detection
                edges = cv2.Canny(frames[i], 100, 200)
                edge_density = np.count_nonzero(edges) / edges.size
                edge_density_scores.append(edge_density)
            
            # Calculate visual metrics
            average_motion = np.mean(motion_scores)
            color_variation = np.mean(color_variation_scores)
            edge_density = np.mean(edge_density_scores)
            
            # Convert frame indices to time for audio correlation
            start_time = 0 / frame_rate
            end_time = len(frames) / frame_rate
            
            # Check for overlapping audio segments
            audio_activity = 0
            for a_start, a_end in audio_segments:
                if a_start < end_time and a_end > start_time:
                    overlap_start = max(start_time, a_start)
                    overlap_end = min(end_time, a_end)
                    audio_activity += overlap_end - overlap_start
            
            audio_activity_ratio = audio_activity / (end_time - start_time)
            
            # Calculate motion consistency
            motion_consistency = np.std(motion_scores) / (np.mean(motion_scores) + 1e-6)
            
            # Combined decision based on multiple factors
            is_action = (
                average_motion > self.action_threshold or  # High motion
                color_variation > 50 or                    # Significant color changes
                audio_activity_ratio > 0.3 or             # Significant audio activity
                (edge_density > 0.15 and 
                 motion_consistency < 0.5)                # Consistent high action
            )
            
            if is_action:
                self.log(
                    f"Action sequence detected: motion={average_motion:.3f}, "
                    f"color_var={color_variation:.3f}, audio_ratio={audio_activity_ratio:.3f}",
                    "INFO", "DETECTION"
                )
            
            return is_action
            
        except Exception as e:
            self.log(f"Error in action sequence detection: {e}", "ERROR", "DETECTION")
            return False

    def analyze_video_section(self, frames: List[np.ndarray],
                            audio_segments: List[Tuple[float, float]],
                            frame_rate: float) -> Dict[str, Any]:
        """Analyze a section of video frames for various characteristics."""
        try:
            # Basic motion and static detection
            is_static = all(
                self.detect_static_frame(frames[i], frames[i+1])
                for i in range(len(frames)-1)
            )
            
            # Menu detection with temporal consistency
            menu_frames = sum(
                1 for frame in frames if self.detect_menu_screen(frame)
            )
            is_menu = menu_frames > len(frames) * 0.5  # More than 50% menu frames
            
            # Activity detection
            is_low_activity = self.detect_low_activity(frames)
            is_action = self.detect_action_sequence(frames, audio_segments, frame_rate)
            
            # Scene complexity analysis
            complexity_scores = []
            for frame in frames:
                # Convert to grayscale
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Calculate various complexity metrics
                laplacian = cv2.Laplacian(gray, cv2.CV_64F)
                gradient_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
                gradient_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
                
                # Combine complexity metrics
                complexity = (
                    0.4 * np.std(laplacian) +
                    0.3 * np.std(gradient_x) +
                    0.3 * np.std(gradient_y)
                )
                complexity_scores.append(complexity)
            
            average_complexity = np.mean(complexity_scores)
            
            # Calculate temporal continuity
            continuity_scores = []
            for i in range(1, len(frames)):
                flow = cv2.calcOpticalFlowFarneback(
                    cv2.cvtColor(frames[i-1], cv2.COLOR_BGR2GRAY),
                    cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY),
                    None, 0.5, 3, 15, 3, 5, 1.2, 0
                )
                continuity_scores.append(np.mean(np.abs(flow)))
            
            temporal_continuity = np.mean(continuity_scores) if continuity_scores else 0
            
            return {
                'is_static': is_static,
                'is_menu': is_menu,
                'is_low_activity': is_low_activity,
                'is_action': is_action,
                'complexity': average_complexity,
                'temporal_continuity': temporal_continuity,
                'menu_frame_ratio': menu_frames / len(frames)
            }
            
        except Exception as e:
            self.log(f"Error in video section analysis: {e}", "ERROR", "ANALYSIS")
            return {
                'is_static': False,
                'is_menu': False,
                'is_low_activity': False,
                'is_action': False,
                'complexity': 0.0,
                'temporal_continuity': 0.0,
                'menu_frame_ratio': 0.0
            }

    def calculate_speed_factor(self, scene_duration: float, total_duration: float,
                             target_duration: float, scene_position: float,
                             scene_analysis: Dict[str, Any]) -> float:
        """Calculate appropriate speed factor for a scene."""
        try:
            speed_settings = self.settings.get('speed_settings', {})
            max_speed = speed_settings.get('max_speed_factor', 4.0)
            min_speed = speed_settings.get('min_speed_factor', 1.0)
            
            # Keep action scenes at normal speed
            if scene_analysis['is_action']:
                return 1.0
            
            # Fast-forward menu screens
            if scene_analysis['is_menu']:
                return min(max_speed, 2.0)
            
            # Calculate required overall speed factor
            overall_speed_needed = total_duration / target_duration
            
            # Calculate dynamic speed factor based on scene position
            # Scenes in the middle can be sped up more
            position_factor = 1.0 + np.sin(scene_position * np.pi) * 0.5
            
            # Adjust for scene complexity
            complexity_factor = 1.0 + (1.0 - min(scene_analysis['complexity'] / 100.0, 1.0))
            
            # Adjust for temporal continuity
            continuity_factor = 1.0 + (1.0 - min(scene_analysis['temporal_continuity'], 1.0))
            
            # Calculate base speed factor
            base_factor = overall_speed_needed * position_factor * complexity_factor * continuity_factor
            
            # Adjust for scene duration - longer scenes can be sped up more
            duration_factor = min(scene_duration / 10.0, 1.0)
            adjusted_factor = base_factor * (1.0 + duration_factor)
            
            # Clamp between min and max speed
            final_factor = max(min_speed, min(max_speed, adjusted_factor))
            
            self.log(
                f"Calculated speed factor: {final_factor:.2f}x for {scene_duration:.2f}s scene "
                f"(complexity={scene_analysis['complexity']:.2f})",
                "DEBUG",
                "SPEED"
            )
            
            return final_factor
            
        except Exception as e:
            self.log(f"Error calculating speed factor: {e}", "ERROR", "SPEED")
            return 1.0

    def analyze_video(self, video_path: str, target_duration: float) -> Dict[str, Any]:
        """Perform complete analysis of video file."""
        try:
            # Create preview for faster analysis
            self.log("Creating preview video for analysis", "INFO", "ANALYSIS")
            preview_path = self.preview_generator.create_preview(video_path)
            if not preview_path:
                raise AnalysisError("Failed to create preview video")
            
            # Extract audio for analysis
            self.log("Extracting and analyzing audio", "INFO", "ANALYSIS")
            audio_data = self.audio_analyzer.extract_audio(video_path)
            audio_segments = self.audio_analyzer.detect_high_activity(audio_data)
            
            # Extract frames from preview
            self.log("Extracting frames for analysis", "INFO", "ANALYSIS")
            frames = list(extract_frames_optimized(preview_path))
            if not frames:
                raise AnalysisError("Failed to extract frames from video")
            
            # Get video information
            cap = cv2.VideoCapture(preview_path)
            frame_rate = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            
            # Detect scenes
            self.log("Detecting scenes", "INFO", "ANALYSIS")
            scenes = self.scene_manager.detect_scenes(frames, audio_segments)
            
            # Analyze each scene
            self.log(f"Analyzing {len(scenes)} detected scenes", "INFO", "ANALYSIS")
            total_processed = 0
            
            for i, scene in enumerate(scenes):
                # Update progress
                progress = (i + 1) / len(scenes) * 100
                self.log(f"Analyzing scene {i+1}/{len(scenes)} ({progress:.1f}%)", "DEBUG", "ANALYSIS")
                
                # Get scene frames
                scene_frames = frames[scene['start_frame']:scene['end_frame']]
                
                # Monitor memory usage
                if monitor_memory_usage():
                    self.log("High memory usage detected - cleaning up", "WARNING", "MEMORY")
                    # Clear frame buffer
                    self.frame_buffer.clear()
                    self.scene_buffer.clear()
                
                # Analyze scene
                analysis = self.analyze_video_section(
                    scene_frames,
                    audio_segments,
                    frame_rate
                )
                scene.update(analysis)
                
                # Calculate speed factor
                scene_duration = (scene['end_frame'] - scene['start_frame']) / frame_rate
                scene_position = (scene['start_frame'] + scene['end_frame']) / (2 * total_frames)
                
                scene['speed_factor'] = self.calculate_speed_factor(
                    scene_duration,
                    total_frames / frame_rate,
                    target_duration,
                    scene_position,
                    analysis
                )
                
                total_processed += scene_duration
            
            # Calculate final statistics
            total_action_time = sum(
                (s['end_frame'] - s['start_frame']) / frame_rate
                for s in scenes if s['is_action']
            )
            
            total_menu_time = sum(
                (s['end_frame'] - s['start_frame']) / frame_rate
                for s in scenes if s['is_menu']
            )
            
            average_complexity = np.mean([s['complexity'] for s in scenes])
            
            stats = {
                'total_scenes': len(scenes),
                'total_duration': total_processed,
                'action_duration': total_action_time,
                'menu_duration': total_menu_time,
                'average_complexity': average_complexity,
                'frame_rate': frame_rate,
                'total_frames': total_frames
            }
            
            self.log(
                f"Completed analysis: {stats['total_scenes']} scenes, "
                f"{stats['action_duration']:.1f}s action, "
                f"{stats['menu_duration']:.1f}s menus",
                "INFO",
                "ANALYSIS"
            )
            
            return {
                'scenes': scenes,
                'stats': stats,
                'frame_rate': frame_rate,
                'total_frames': total_frames
            }
            
        except Exception as e:
            error_msg = f"Error in video analysis: {e}"
            self.log(error_msg, "ERROR", "ANALYSIS")
            return {
                'scenes': [],
                'stats': {},
                'frame_rate': 0,
                'total_frames': 0
            }

# Create analyzer instance
analyzer = VideoAnalyzer()

def analyze_video_file(video_path: str, target_duration: float) -> Dict[str, Any]:
    """Analyze a single video file."""
    try:
        return analyzer.analyze_video(video_path, target_duration)
    except Exception as e:
        log_event(f"Error analyzing video file: {e}", "ERROR", "ANALYSIS")
        return {
            'scenes': [],
            'stats': {},
            'frame_rate': 0,
            'total_frames': 0
        }

def estimate_processing_time(video_path: str) -> float:
    """Estimate the time needed to analyze the video."""
    try:
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        duration = frame_count / fps
        cap.release()

        # Base time calculation (empirical formula)
        base_time = duration * 0.1  # Approximately 10% of video duration
        
        # Adjust for resolution
        cap = cv2.VideoCapture(video_path)
        _, frame = cap.read()
        if frame is not None:
            resolution_factor = (frame.shape[0] * frame.shape[1]) / (1280 * 720)
            base_time *= min(resolution_factor, 2.0)
        cap.release()

        return base_time
    except Exception as e:
        log_event(f"Error estimating processing time: {e}", "ERROR", "ANALYSIS")
        return 0.0

def get_video_statistics(video_path: str) -> Dict[str, Any]:
    """Get basic statistics about the video file."""
    try:
        cap = cv2.VideoCapture(video_path)
        stats = {
            'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'fps': int(cap.get(cv2.CAP_PROP_FPS)),
            'duration': cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS),
            'size': os.path.getsize(video_path),
            'resolution': (
                int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            )
        }
        cap.release()
        return stats
    except Exception as e:
        log_event(f"Error getting video statistics: {e}", "ERROR", "ANALYSIS")
        return {}

def validate_analysis_results(results: Dict[str, Any]) -> bool:
    """Validate the analysis results for completeness and consistency."""
    try:
        # Check for required keys
        required_keys = ['scenes', 'stats', 'frame_rate', 'total_frames']
        if not all(key in results for key in required_keys):
            return False

        # Check scene data consistency
        if not results['scenes']:
            return False

        prev_end = 0
        for scene in results['scenes']:
            # Check scene has required properties
            required_scene_keys = [
                'start_frame', 'end_frame', 'is_action',
                'is_menu', 'is_static', 'speed_factor'
            ]
            if not all(key in scene for key in required_scene_keys):
                return False

            # Check frame boundaries
            if (scene['start_frame'] >= scene['end_frame'] or
                scene['start_frame'] < prev_end or
                scene['end_frame'] > results['total_frames']):
                return False

            prev_end = scene['end_frame']

        # Check statistics
        stats = results['stats']
        required_stats = [
            'total_scenes', 'total_duration', 'action_duration',
            'menu_duration', 'average_complexity'
        ]
        if not all(key in stats for key in required_stats):
            return False

        return True
    except Exception as e:
        log_event(f"Error validating analysis results: {e}", "ERROR", "ANALYSIS")
        return False

if __name__ == "__main__":
    # Example usage and testing
    import sys
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
        if os.path.exists(video_path):
            print(f"Analyzing video: {video_path}")
            stats = get_video_statistics(video_path)
            print("\nVideo Statistics:")
            for key, value in stats.items():
                print(f"{key}: {value}")
                
            est_time = estimate_processing_time(video_path)
            print(f"\nEstimated processing time: {est_time:.1f} seconds")
            
            target_duration = stats['duration'] * 0.5  # Target 50% of original duration
            results = analyze_video_file(video_path, target_duration)
            
            if validate_analysis_results(results):
                print("\nAnalysis completed successfully!")
                print(f"Detected {len(results['scenes'])} scenes")
                print(f"Action sequences: {results['stats']['action_duration']:.1f} seconds")
                print(f"Menu sequences: {results['stats']['menu_duration']:.1f} seconds")
            else:
                print("\nError: Analysis results validation failed")
        else:
            print(f"Error: File not found - {video_path}")
    else:
        print("Usage: python analyze.py <video_file>")
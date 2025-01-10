# .\scripts\generate.py

import os, sys, cv2, numpy as np, moviepy.editor as mp
from typing import List, Dict, Any, Optional, Tuple
from utility import (
    analyze_segment,
    extract_frames,
    load_settings,
    log_event,
    cleanup_work_directory
)

class VideoProcessor:
    def __init__(self):
        self.settings = load_settings()
        self.search_criteria = self.settings.get('search', {})
        self.video_config = self.settings.get('video', {})
        self.work_dir = self.video_config.get('temp_directory', 'work')
        
    def process_videos(self, input_dir: str, output_dir: str) -> None:
        """Process all videos in input directory."""
        try:
            video_files = [
                f for f in os.listdir(input_dir)
                if os.path.splitext(f)[1].lower() in self.video_config.get('supported_formats', [])
            ]
            
            if not video_files:
                log_event("No video files found in input directory")
                return
                
            total_files = len(video_files)
            total_duration = self._calculate_total_duration(input_dir, video_files)
            log_event(f"Starting processing of {total_files} videos. Total duration: {total_duration:.2f} seconds")
            
            # Phase 1: Create preview versions
            preview_files = self._create_preview_versions(input_dir, video_files)
            
            # Phase 2: Process static content and menus
            processed_files_p2 = self._process_phase2(preview_files)
            
            # Phase 3: Process low activity sections
            processed_files_p3 = self._process_phase3(processed_files_p2)
            
            # Phase 4: Scene detection and speed adjustment
            processed_files_p4 = self._process_phase4(processed_files_p3, total_duration)
            
            # Phase 5: Final compilation
            final_output = self._compile_final_video(processed_files_p4, output_dir)
            
            cleanup_work_directory()
            log_event("Video processing complete")
            
        except Exception as e:
            log_event(f"Error in process_videos: {e}")
            cleanup_work_directory()

    def _calculate_total_duration(self, input_dir: str, video_files: List[str]) -> float:
        """Calculate total duration of all input videos."""
        total_duration = 0
        for video_file in video_files:
            video_path = os.path.join(input_dir, video_file)
            try:
                clip = mp.VideoFileClip(video_path)
                duration = clip.duration
                size_mb = os.path.getsize(video_path) / (1024 * 1024)
                log_event(f"Video: {video_file}, Duration: {duration:.2f}s, Size: {size_mb:.2f}MB")
                total_duration += duration
                clip.close()
            except Exception as e:
                log_event(f"Error calculating duration for {video_file}: {e}")
        return total_duration

    def _create_preview_versions(self, input_dir: str, video_files: List[str]) -> List[str]:
        """Create 360p preview versions of all videos."""
        preview_files = []
        for video_file in video_files:
            input_path = os.path.join(input_dir, video_file)
            preview_path = os.path.join(self.work_dir, f"preview_{video_file}")
            try:
                clip = mp.VideoFileClip(input_path)
                # Calculate new dimensions maintaining aspect ratio
                aspect_ratio = clip.w / clip.h
                new_height = 360
                new_width = int(new_height * aspect_ratio)
                # Resize and write preview
                preview = clip.resize(height=new_height, width=new_width)
                preview.write_videofile(preview_path, codec='libx264')
                preview_files.append(preview_path)
                clip.close()
                preview.close()
                log_event(f"Created preview version: {preview_path}")
            except Exception as e:
                log_event(f"Error creating preview for {video_file}: {e}")
        return preview_files

    def _detect_static_frame(self, frame1: np.ndarray, frame2: np.ndarray, threshold: float = 0.99) -> bool:
        """Detect if frames are static (nearly identical)."""
        try:
            diff = cv2.absdiff(frame1, frame2)
            diff_score = 1 - (np.count_nonzero(diff) / diff.size)
            return diff_score > threshold
        except Exception as e:
            log_event(f"Error in static frame detection: {e}")
            return False

    def _detect_menu_screen(self, frame: np.ndarray) -> bool:
        """Detect if frame is likely a menu screen."""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Look for text-like features
            edges = cv2.Canny(gray, 50, 150)
            # Calculate edge density
            edge_density = np.count_nonzero(edges) / edges.size
            # Menu screens typically have high edge density
            return edge_density > 0.1
        except Exception as e:
            log_event(f"Error in menu detection: {e}")
            return False

    def _process_phase2(self, preview_files: List[str]) -> List[str]:
        """Process static content and menu screens."""
        processed_files = []
        for preview_file in preview_files:
            output_path = os.path.join(self.work_dir, f"p2_{os.path.basename(preview_file)}")
            try:
                # Extract frames
                cap = cv2.VideoCapture(preview_file)
                frames = []
                static_segments = []
                menu_segments = []
                
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frames.append(frame)
                
                # Detect static segments and menus
                for i in range(1, len(frames)):
                    if self._detect_static_frame(frames[i-1], frames[i]):
                        static_segments.append(i)
                    if self._detect_menu_screen(frames[i]):
                        menu_segments.append(i)
                
                cap.release()
                
                # Process video with detected segments
                clip = mp.VideoFileClip(preview_file)
                # Remove static segments
                # Speed up menu segments
                clip.write_videofile(output_path, codec='libx264')
                processed_files.append(output_path)
                clip.close()
                
            except Exception as e:
                log_event(f"Error in phase 2 processing for {preview_file}: {e}")
        
        return processed_files

    def _detect_low_activity(self, frames: List[np.ndarray], threshold: float = 0.1) -> bool:
        """Detect sections with low activity."""
        try:
            motion_scores = []
            for i in range(1, len(frames)):
                diff = cv2.absdiff(frames[i-1], frames[i])
                motion_score = np.mean(diff) / 255.0
                motion_scores.append(motion_score)
            return np.mean(motion_scores) < threshold
        except Exception as e:
            log_event(f"Error in low activity detection: {e}")
            return False

    def _process_phase3(self, phase2_files: List[str]) -> List[str]:
        """Process low activity sections."""
        processed_files = []
        for input_file in phase2_files:
            output_path = os.path.join(self.work_dir, f"p3_{os.path.basename(input_file)}")
            try:
                frames = extract_frames(input_file)
                low_activity_segments = []
                
                # Analyze in windows
                window_size = 30  # 1 second at 30fps
                for i in range(0, len(frames), window_size):
                    window = frames[i:i+window_size]
                    if len(window) == window_size and self._detect_low_activity(window):
                        low_activity_segments.append((i, i+window_size))
                
                # Process video with detected segments
                clip = mp.VideoFileClip(input_file)
                # Speed up low activity segments
                clip.write_videofile(output_path, codec='libx264')
                processed_files.append(output_path)
                clip.close()
                
            except Exception as e:
                log_event(f"Error in phase 3 processing for {input_file}: {e}")
        
        return processed_files

    def _detect_scene_change(self, frame1: np.ndarray, frame2: np.ndarray, threshold: float = 30.0) -> bool:
        """Detect scene changes between frames."""
        try:
            hist1 = cv2.calcHist([frame1], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            hist2 = cv2.calcHist([frame2], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            diff = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CHISQR)
            return diff > threshold
        except Exception as e:
            log_event(f"Error in scene change detection: {e}")
            return False

    def _detect_action_sequence(self, frames: List[np.ndarray], threshold: float = 0.3) -> bool:
        """Detect action sequences based on motion and color changes."""
        try:
            motion_scores = []
            for i in range(1, len(frames)):
                # Motion detection
                diff = cv2.absdiff(frames[i-1], frames[i])
                motion_score = np.mean(diff) / 255.0
                motion_scores.append(motion_score)
            
            # High motion scores indicate action
            return np.mean(motion_scores) > threshold
        except Exception as e:
            log_event(f"Error in action sequence detection: {e}")
            return False

    def _process_phase4(self, phase3_files: List[str], target_duration: float) -> List[str]:
        """Scene detection and speed adjustment."""
        processed_files = []
        for input_file in phase3_files:
            output_path = os.path.join(self.work_dir, f"p4_{os.path.basename(input_file)}")
            try:
                frames = extract_frames(input_file)
                scenes = []
                current_scene_start = 0
                
                # Detect scenes and action sequences
                for i in range(1, len(frames)):
                    if self._detect_scene_change(frames[i-1], frames[i]):
                        scenes.append({
                            'start': current_scene_start,
                            'end': i,
                            'action': self._detect_action_sequence(frames[current_scene_start:i])
                        })
                        current_scene_start = i
                
                # Process scenes with variable speed
                clip = mp.VideoFileClip(input_file)
                final_clips = []
                
                for scene in scenes:
                    scene_clip = clip.subclip(scene['start']/30, scene['end']/30)  # Convert frame numbers to seconds
                    if scene['action']:
                        # Keep action sequences at normal speed
                        final_clips.append(scene_clip)
                    else:
                        # Apply variable speed to non-action scenes
                        duration = scene_clip.duration
                        speed_factor = self._calculate_speed_factor(duration, target_duration, len(scenes))
                        final_clips.append(scene_clip.speedx(speed_factor))
                
                # Concatenate scenes
                final_clip = mp.concatenate_videoclips(final_clips)
                final_clip.write_videofile(output_path, codec='libx264')
                processed_files.append(output_path)
                
                clip.close()
                final_clip.close()
                for c in final_clips:
                    c.close()
                
            except Exception as e:
                log_event(f"Error in phase 4 processing for {input_file}: {e}")
        
        return processed_files

    def _calculate_speed_factor(self, scene_duration: float, target_duration: float, total_scenes: int) -> float:
        """Calculate speed factor for scene based on target duration."""
        try:
            # Base speed factor on scene duration and target duration
            return max(1.0, scene_duration / (target_duration / total_scenes))
        except Exception as e:
            log_event(f"Error calculating speed factor: {e}")
            return 1.0

    def _compile_final_video(self, processed_files: List[str], output_dir: str) -> str:
        """Compile all processed videos into final output."""
        final_output = os.path.join(output_dir, "final_output.mp4")
        try:
            clips = [mp.VideoFileClip(f) for f in processed_files]
            final_clip = mp.concatenate_videoclips(clips)
            final_clip.write_videofile(final_output, codec='libx264')
            
            # Cleanup
            final_clip.close()
            for clip in clips:
                clip.close()
            
            log_event(f"Final video compiled: {final_output}")
            return final_output
            
        except Exception as e:
            log_event(f"Error compiling final video: {e}")
            return ""

# Create processor instance
processor = VideoProcessor()

def process_videos(input_dir: str, output_dir: str) -> None:
    """Main entry point for video processing."""
    processor.process_videos(input_dir, output_dir)

def process_video(input_path: str, output_path: str) -> Optional[str]:
    """Process a single video file."""
    return processor.process_video(input_path, output_path)
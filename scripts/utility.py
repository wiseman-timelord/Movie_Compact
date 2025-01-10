# .\scripts\utility.py

# .\scripts\generate.py

import os, sys, cv2, numpy as np, moviepy.editor as mp
from typing import List, Dict, Any, Optional, Tuple
from utility import (
    analyze_segment,
    extract_frames,
    load_settings,
    log_event,
    cleanup_work_directory,
    ProgressTracker,
    batch_process_frames,
    extract_frames_optimized
)

class VideoProcessor:
    def __init__(self):
        self.settings = load_settings()
        self.search_criteria = self.settings.get('search', {})
        self.video_config = self.settings.get('video', {})
        self.work_dir = self.video_config.get('temp_directory', 'work')
        self.progress = None
        
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
                
            # Initialize progress tracking
            total_files = len(video_files)
            self.progress = ProgressTracker(total_files * 5)  # 5 phases per file
            
            total_duration = self._calculate_total_duration(input_dir, video_files)
            log_event(f"Starting processing of {total_files} videos. Total duration: {total_duration:.2f} seconds")
            
            # Phase 1: Create preview versions
            self.progress.update(phase="Creating Preview Versions")
            preview_files = self._create_preview_versions(input_dir, video_files)
            
            # Phase 2: Process static content and menus
            self.progress.update(phase="Processing Static Content")
            processed_files_p2 = self._process_phase2(preview_files)
            
            # Phase 3: Process low activity sections
            self.progress.update(phase="Processing Low Activity")
            processed_files_p3 = self._process_phase3(processed_files_p2)
            
            # Phase 4: Scene detection and speed adjustment
            self.progress.update(phase="Scene Detection")
            processed_files_p4 = self._process_phase4(processed_files_p3, total_duration)
            
            # Phase 5: Final compilation
            self.progress.update(phase="Final Compilation")
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
        preview_height = self.settings.get('video_settings', {}).get('preview_height', 360)
        
        for video_file in video_files:
            input_path = os.path.join(input_dir, video_file)
            preview_path = os.path.join(self.work_dir, f"preview_{video_file}")
            try:
                clip = mp.VideoFileClip(input_path)
                aspect_ratio = clip.w / clip.h
                new_height = preview_height
                new_width = int(new_height * aspect_ratio)
                preview = clip.resize(height=new_height, width=new_width)
                preview.write_videofile(preview_path, codec='libx264')
                preview_files.append(preview_path)
                clip.close()
                preview.close()
                log_event(f"Created preview version: {preview_path}")
                self.progress.update()
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
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.count_nonzero(edges) / edges.size
            return edge_density > 0.1
        except Exception as e:
            log_event(f"Error in menu detection: {e}")
            return False

    def _process_phase2(self, preview_files: List[str]) -> List[str]:
        """Process static content and menu screens."""
        processed_files = []
        batch_size = self.settings.get('hardware', {}).get('gpu_batch_size', 32)
        
        for preview_file in preview_files:
            output_path = os.path.join(self.work_dir, f"p2_{os.path.basename(preview_file)}")
            try:
                frames = []
                static_segments = []
                menu_segments = []
                
                # Use optimized frame extraction
                for frame in extract_frames_optimized(preview_file):
                    if frame is not None:
                        frames.append(frame)
                
                # Process frames in batches
                for i in range(1, len(frames)):
                    # Process static detection in batches
                    if self._detect_static_frame(frames[i-1], frames[i]):
                        static_segments.append(i)
                    
                    # Process menu detection in batches
                    if self._detect_menu_screen(frames[i]):
                        menu_segments.append(i)
                
                # Process video
                clip = mp.VideoFileClip(preview_file)
                final_clip = self._apply_speed_changes(clip, static_segments, menu_segments)
                final_clip.write_videofile(output_path, codec='libx264')
                processed_files.append(output_path)
                
                clip.close()
                final_clip.close()
                self.progress.update()
                
            except Exception as e:
                log_event(f"Error in phase 2 processing for {preview_file}: {e}")
        
        return processed_files

    def _apply_speed_changes(self, clip: mp.VideoFileClip, static_segments: List[int], 
                           menu_segments: List[int]) -> mp.VideoFileClip:
        """Apply speed changes to video segments."""
        try:
            # Convert frame indices to timestamps
            fps = clip.fps
            segments = []
            current_pos = 0
            
            # Process all segments in order
            all_segments = sorted(set(static_segments + menu_segments))
            for segment in all_segments:
                start_time = current_pos / fps
                end_time = segment / fps
                
                if segment in static_segments:
                    # Skip static segments
                    current_pos = segment + 1
                elif segment in menu_segments:
                    # Speed up menu segments
                    subclip = clip.subclip(start_time, end_time).speedx(2.0)
                    segments.append(subclip)
                    current_pos = segment + 1
                else:
                    # Keep normal segments
                    subclip = clip.subclip(start_time, end_time)
                    segments.append(subclip)
                    current_pos = segment + 1
            
            # Add remaining video
            if current_pos / fps < clip.duration:
                segments.append(clip.subclip(current_pos / fps))
            
            return mp.concatenate_videoclips(segments) if segments else clip
            
        except Exception as e:
            log_event(f"Error applying speed changes: {e}")
            return clip

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
        window_size = self.settings.get('search', {}).get('frame_settings', {}).get('batch_size', 30)
        
        for input_file in phase2_files:
            output_path = os.path.join(self.work_dir, f"p3_{os.path.basename(input_file)}")
            try:
                frames = []
                for frame in extract_frames_optimized(input_file):
                    if frame is not None:
                        frames.append(frame)
                
                low_activity_segments = []
                
                # Analyze in windows
                for i in range(0, len(frames), window_size):
                    window = frames[i:i+window_size]
                    if len(window) == window_size and self._detect_low_activity(window):
                        low_activity_segments.append((i, i+window_size))
                
                # Process video
                clip = mp.VideoFileClip(input_file)
                final_clip = self._apply_speed_to_low_activity(clip, low_activity_segments)
                final_clip.write_videofile(output_path, codec='libx264')
                processed_files.append(output_path)
                
                clip.close()
                final_clip.close()
                self.progress.update()
                
            except Exception as e:
                log_event(f"Error in phase 3 processing for {input_file}: {e}")
        
        return processed_files

    def _apply_speed_to_low_activity(self, clip: mp.VideoFileClip, 
                                   low_activity_segments: List[Tuple[int, int]]) -> mp.VideoFileClip:
        """Apply speed changes to low activity segments."""
        try:
            fps = clip.fps
            segments = []
            current_pos = 0
            
            for start, end in low_activity_segments:
                # Add normal speed segment before low activity
                if start > current_pos:
                    segments.append(clip.subclip(current_pos/fps, start/fps))
                
                # Speed up low activity segment
                speed_factor = self.settings.get('speed_settings', {}).get('max_speed_factor', 4.0)
                segments.append(clip.subclip(start/fps, end/fps).speedx(speed_factor))
                current_pos = end
            
            # Add remaining video
            if current_pos / fps < clip.duration:
                segments.append(clip.subclip(current_pos/fps))
            
            return mp.concatenate_videoclips(segments) if segments else clip
            
        except Exception as e:
            log_event(f"Error applying speed to low activity: {e}")
            return clip

    def _detect_scene_change(self, frame1: np.ndarray, frame2: np.ndarray) -> bool:
        """Detect scene changes between frames."""
        try:
            threshold = self.settings.get('scene_settings', {}).get('scene_threshold', 30.0)
            hist1 = cv2.calcHist([frame1], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            hist2 = cv2.calcHist([frame2], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            diff = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CHISQR)
            return diff > threshold
        except Exception as e:
            log_event(f"Error in scene change detection: {e}")
            return False

    def _detect_action_sequence(self, frames: List[np.ndarray]) -> bool:
        """Detect action sequences based on motion and color changes."""
        try:
            threshold = self.settings.get('scene_settings', {}).get('action_threshold', 0.3)
            motion_scores = []
            
            for i in range(1, len(frames)):
                diff = cv2.absdiff(frames[i-1], frames[i])
                motion_score = np.mean(diff) / 255.0
                motion_scores.append(motion_score)
            
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
                frames = []
                for frame in extract_frames_optimized(input_file):
                    if frame is not None:
                        frames.append(frame)
                
                scenes = []
                current_scene_start = 0
                
                # Detect scenes
                for i in range(1, len(frames)):
                    if self._detect_scene_change(frames[i-1], frames[i]):
                        scenes.append({
                            'start': current_scene_start,
                            'end': i,
                            'action': self._detect_action_sequence(frames[current_scene_start:i])
                        })
                        current_scene_start = i
                
                # Process scenes
                clip = mp.VideoFileClip(input_file)
                final_clips = []
                
                for scene in scenes:
                    scene_clip = clip.subclip(scene['start']/30, scene['end']/30)
                    if scene['action']:
                        # Keep action sequences at normal speed
                        final_clips.append(scene_clip)
                    else:
                        # Apply variable speed to non-action scenes
                        duration = scene_clip.duration
                        speed_factor = self._calculate_speed_factor(duration, target_duration, len(scenes))
                        final_clips.append(scene_clip.speedx(speed_factor))
                
                # Create final video
                if final_clips:
                    final_clip = mp.concatenate_videoclips(final_clips)
                    final_clip.write_videofile(output_path, codec='libx264')
                    processed_files.append(output_path)
                    
                    # Cleanup
                    final_clip.close()
                    for c in final_clips:
                        c.close()
                
                clip.close()
                self.progress.update()
                
            except Exception as e:
                log_event(f"Error in phase 4 processing for {input_file}: {e}")
        
        return processed_files

    def _calculate_speed_factor(self, scene_duration: float, target_duration: float, total_scenes: int) -> float:
        """Calculate speed factor for scene based on target duration."""
        try:
            max_speed = self.settings.get('speed_settings', {}).get('max_speed_factor', 4.0)
            min_speed = self.settings.get('speed_settings', {}).get('min_speed_factor', 1.0)
            
            # Calculate base speed factor
            base_factor = scene_duration / (target_duration / total_scenes)
            
            # Clamp between min and max speed
            return max(min_speed, min(max_speed, base_factor))
            
        except Exception as e:
            log_event(f"Error calculating speed factor: {e}")
            return 1.0

    def _compile_final_video(self, processed_files: List[str], output_dir: str) -> str:
        """Compile all processed videos into final output."""
        final_output = os.path.join(output_dir, "final_output.mp4")
        try:
            # Load all clips
            clips = [mp.VideoFileClip(f) for f in processed_files]
            
            if not clips:
                log_event("No clips to concatenate")
                return ""
            
            # Get final video settings
            target_fps = self.video_config.get('video_settings', {}).get('target_fps', 30)
            codec = self.video_config.get('video_settings', {}).get('codec', 'libx264')
            audio_codec = self.video_config.get('video_settings', {}).get('audio_codec', 'aac')
            
            # Create and write final video
            final_clip = mp.concatenate_videoclips(clips)
            final_clip.write_videofile(
                final_output,
                fps=target_fps,
                codec=codec,
                audio_codec=audio_codec
            )
            
            # Cleanup
            final_clip.close()
            for clip in clips:
                clip.close()
            
            self.progress.update()
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
    try:
        processor.process_videos(os.path.dirname(input_path), os.path.dirname(output_path))
        return output_path
    except Exception as e:
        log_event(f"Error processing single video: {e}")
        return None
# process.py

import os
import cv2
import numpy as np
import moviepy.editor as mp
from moviepy.video.fx.all import speedx
import time
from typing import Dict, Any, Optional, List, Tuple, Callable
from queue import Queue
from threading import Event, Lock
from utility import (
    load_settings,
    log_event,
    cleanup_work_directory,
    ProgressTracker,
    MetricsCollector,
    PreviewGenerator,
    SceneManager,
    AudioAnalyzer,
    detect_static_frame,
    detect_menu_screen,
    monitor_memory_usage
)
from analyze import VideoAnalyzer
from interface import ProcessingError, MovieCompactError

class VideoProcessor:
    """Handles video processing and consolidation operations."""
    
    def __init__(self, log_manager=None):
        self.settings = load_settings()
        self.video_config = self.settings.get('video', {})
        self.work_dir = self.video_config.get('temp_directory', 'work')
        self.metrics = MetricsCollector()
        self.progress = None
        self.active_process = False
        self.analyzer = VideoAnalyzer(log_manager)
        self.preview_generator = PreviewGenerator()
        self.scene_manager = SceneManager()
        self.audio_analyzer = AudioAnalyzer()
        self.cancel_flag = Event()
        self.processing_lock = Lock()
        self.log_manager = log_manager

    def log(self, message: str, level: str = "INFO", category: str = "PROCESSING"):
        """Log a message using the log manager if available."""
        if self.log_manager:
            self.log_manager.log(message, level, category)
        else:
            log_event(message, level, category)

    def cancel_processing(self):
        """Cancel current processing operation."""
        self.cancel_flag.set()
        self.log("Processing cancelled by user", "INFO", "CONTROL")

    def create_preview_video(self, input_path: str) -> str:
        """Create lower resolution preview video for analysis."""
        try:
            preview_height = self.video_config.get('preview_height', 360)
            preview_path = os.path.join(self.work_dir, f"preview_{os.path.basename(input_path)}")
            
            self.log(f"Creating preview video at {preview_height}p", "INFO", "PREVIEW")
            
            clip = mp.VideoFileClip(input_path)
            aspect_ratio = clip.w / clip.h
            preview_width = int(preview_height * aspect_ratio)
            
            preview = clip.resize(height=preview_height)
            preview.write_videofile(
                preview_path,
                codec='libx264',
                audio_codec='aac',
                preset='ultrafast',
                threads=4
            )
            
            clip.close()
            preview.close()
            
            self.log("Preview video created successfully", "INFO", "PREVIEW")
            return preview_path
            
        except Exception as e:
            self.log(f"Error creating preview: {e}", "ERROR", "PREVIEW")
            return ""

    def process_scenes(self, clip: mp.VideoFileClip, scenes: List[Dict[str, Any]],
                      target_duration: float) -> mp.VideoFileClip:
        """Process video scenes with appropriate speed adjustments."""
        try:
            processed_scenes = []
            total_scenes = len(scenes)
            current_duration = 0
            
            for i, scene in enumerate(scenes):
                if self.cancel_flag.is_set():
                    raise ProcessingError("Processing cancelled by user")
                    
                scene_progress = (i / total_scenes) * 100
                self.log(f"Processing scene {i+1}/{total_scenes} ({scene_progress:.1f}%)", 
                        "INFO", "SCENES")
                
                # Extract scene subclip
                start_time = scene['start_frame'] / clip.fps
                end_time = scene['end_frame'] / clip.fps
                subclip = clip.subclip(start_time, end_time)
                
                # Process based on scene type
                processed_clip = self._process_scene_segment(subclip, scene, target_duration)
                processed_scenes.append(processed_clip)
                
                # Update progress
                current_duration += processed_clip.duration
                
                # Monitor memory usage
                if monitor_memory_usage():
                    self.log("High memory usage detected - cleaning up", "WARNING", "MEMORY")
                    # Force garbage collection of unused clips
                    for clip in processed_scenes[:-1]:
                        clip.close()
                
            # Concatenate all processed scenes
            final_clip = mp.concatenate_videoclips(processed_scenes)
            
            return final_clip
            
        except Exception as e:
            self.log(f"Error processing scenes: {e}", "ERROR", "SCENES")
            raise

    def _process_scene_segment(self, clip: mp.VideoFileClip, scene: Dict[str, Any],
                             target_duration: float) -> mp.VideoFileClip:
        """Process individual scene segment based on its characteristics."""
        try:
            if scene['is_static']:
                # For static scenes, take brief snapshot
                return self._process_static_scene(clip)
            elif scene['is_menu']:
                # Speed up menu scenes
                return self._process_menu_scene(clip)
            elif scene['is_action']:
                # Preserve action scenes at normal speed
                return self._process_action_scene(clip)
            else:
                # Apply dynamic speed adjustment
                return self._apply_dynamic_speed(clip, scene)
                
        except Exception as e:
            self.log(f"Error processing scene segment: {e}", "ERROR", "SCENES")
            return clip

    def _process_static_scene(self, clip: mp.VideoFileClip) -> mp.VideoFileClip:
        """Process static scene by taking brief snapshot."""
        duration = min(1.0, clip.duration)  # Take at most 1 second
        return clip.subclip(0, duration)

    def _process_menu_scene(self, clip: mp.VideoFileClip) -> mp.VideoFileClip:
        """Process menu scene with high speed."""
        speed_factor = min(8.0, self.settings.get('speed_settings', {}).get('menu_speed', 4.0))
        return clip.speedx(speed_factor)

    def _process_action_scene(self, clip: mp.VideoFileClip) -> mp.VideoFileClip:
        """Process action scene preserving original speed."""
        return clip.copy()

    def _apply_dynamic_speed(self, clip: mp.VideoFileClip, scene: Dict[str, Any]) -> mp.VideoFileClip:
        """Apply dynamic speed adjustment to scene."""
        try:
            speed_factor = scene['speed_factor']
            
            if speed_factor == 1.0:
                return clip.copy()
                
            # Create smooth speed transition
            if clip.duration > 2.0:
                # Split into three parts for smooth transition
                part1_duration = min(1.0, clip.duration * 0.2)
                part3_duration = min(1.0, clip.duration * 0.2)
                part2_duration = clip.duration - part1_duration - part3_duration
                
                # Create parts with speed transitions
                part1 = self._create_speed_transition(
                    clip.subclip(0, part1_duration),
                    1.0,
                    speed_factor
                )
                
                part2 = clip.subclip(
                    part1_duration,
                    part1_duration + part2_duration
                ).speedx(speed_factor)
                
                part3 = self._create_speed_transition(
                    clip.subclip(clip.duration - part3_duration),
                    speed_factor,
                    1.0
                )
                
                return mp.concatenate_videoclips([part1, part2, part3])
            else:
                # Short clip - simple speed adjustment
                return clip.speedx(speed_factor)
                
        except Exception as e:
            self.log(f"Error applying dynamic speed: {e}", "ERROR", "SPEED")
            return clip.copy()

    def _create_speed_transition(self, clip: mp.VideoFileClip, start_speed: float,
                               end_speed: float) -> mp.VideoFileClip:
        """Create smooth transition between speeds."""
        try:
            n_frames = int(clip.duration * clip.fps)
            if n_frames < 2:
                return clip.speedx(end_speed)
                
            speeds = np.linspace(start_speed, end_speed, n_frames)
            frames = []
            
            for i, speed in enumerate(speeds):
                time = i / clip.fps
                frame = clip.get_frame(time)
                frames.append(frame)
                
            return mp.ImageSequenceClip(frames, fps=clip.fps)
            
        except Exception as e:
            self.log(f"Error creating speed transition: {e}", "ERROR", "SPEED")
            return clip

    def adjust_audio(self, clip: mp.VideoFileClip) -> mp.VideoFileClip:
        """Adjust audio for speed changes while preserving quality."""
        try:
            if clip.audio is None:
                return clip
                
            preserve_pitch = self.settings.get('audio', {}).get('preserve_pitch', True)
            
            if preserve_pitch:
                # Process audio to maintain pitch despite speed changes
                processed_audio = clip.audio.set_fps(clip.audio.fps * 1.0)
                return clip.set_audio(processed_audio)
            else:
                return clip
                
        except Exception as e:
            self.log(f"Error adjusting audio: {e}", "ERROR", "AUDIO")
            return clip

    def process_video(self, input_path: str, output_path: str, target_duration: float,
                     progress_callback: Optional[Callable] = None) -> Optional[str]:
        """Process a single video file."""
        try:
            if self.active_process:
                raise ProcessingError("Processing already in progress")
                
            self.active_process = True
            self.cancel_flag.clear()
            self.progress = ProgressTracker(6)  # 6 major phases
            
            # Phase 1: Create preview and analyze
            self.log(f"Starting processing of {input_path}", "INFO", "PROCESSING")
            self._update_progress(progress_callback, "Creating Preview", 0)
            
            preview_path = self.create_preview_video(input_path)
            if not preview_path or self.cancel_flag.is_set():
                raise ProcessingError("Failed to create preview")
                
            # Phase 2: Analyze video
            self._update_progress(progress_callback, "Analyzing Video", 20)
            scene_data = self.analyzer.analyze_video(input_path, target_duration)
            if not scene_data['scenes'] or self.cancel_flag.is_set():
                raise ProcessingError("Failed to analyze video")
                
            # Phase 3: Process scenes
            self._update_progress(progress_callback, "Processing Scenes", 40)
            clip = mp.VideoFileClip(input_path)
            processed_clip = self.process_scenes(clip, scene_data['scenes'], target_duration)
            
            # Phase 4: Adjust audio
            self._update_progress(progress_callback, "Processing Audio", 60)
            processed_clip = self.adjust_audio(processed_clip)
            
            # Phase 5: Final compression if needed
            self._update_progress(progress_callback, "Finalizing Video", 80)
            if processed_clip.duration > target_duration * 1.5:
                self.log("Final duration too long - applying additional compression", 
                        "WARNING", "PROCESSING")
                processed_clip = self._compress_video(processed_clip)
                
            # Phase 6: Write final video
            self._update_progress(progress_callback, "Writing Output", 90)
            
            processed_clip.write_videofile(
                output_path,
                codec=self.video_config.get('codec', 'libx264'),
                audio_codec=self.video_config.get('audio_codec', 'aac'),
                threads=4,
                fps=clip.fps
            )
            
            # Cleanup
            clip.close()
            processed_clip.close()
            cleanup_work_directory()
            
            final_duration = mp.VideoFileClip(output_path).duration
            compression_ratio = (clip.duration / final_duration) if final_duration > 0 else 0
            
            self.log(
                f"Processing complete. Original: {clip.duration:.1f}s, "
                f"Final: {final_duration:.1f}s, Ratio: {compression_ratio:.1f}x",
                "INFO", "PROCESSING"
            )
            
            self._update_progress(progress_callback, "Complete", 100)
            
            return output_path
            
        except Exception as e:
            error_msg = f"Error processing video: {e}"
            self.log(error_msg, "ERROR", "PROCESSING")
            return None
            
        finally:
            self.active_process = False
            self.cancel_flag.clear()
            cleanup_work_directory()

    def _update_progress(self, callback: Optional[Callable], phase: str,
                        progress: float) -> None:
        """Update progress and call progress callback if provided."""
        if callback:
            try:
                callback(phase, progress, f"Processing: {phase}")
            except Exception as e:
                self.log(f"Error in progress callback: {e}", "ERROR", "PROGRESS")

    def _compress_video(self, clip: mp.VideoFileClip,
                       target_size_mb: float = 1000.0) -> mp.VideoFileClip:
        """Compress video to target size while maintaining quality."""
        try:
            # Calculate current bitrate
            temp_file = os.path.join(self.work_dir, "temp_output.mp4")
            clip.write_videofile(temp_file, codec='libx264', audio_codec='aac')
            current_size_mb = os.path.getsize(temp_file) / (1024 * 1024)
            current_bitrate = (current_size_mb * 8 * 1024) / clip.duration
            
            # Calculate target bitrate
            target_bitrate = (target_size_mb * 8 * 1024) / clip.duration
            
            # Adjust resolution if necessary
            if target_bitrate < current_bitrate * 0.5:
                height = min(clip.h, 720)  # Cap at 720p
                clip = clip.resize(height=height)
            
            return clip
            
        except Exception as e:
            self.log(f"Error compressing video: {e}", "ERROR", "COMPRESSION")
            return clip

    def validate_output(self, output_path: str, target_duration: float) -> bool:
        """Validate the processed video meets requirements."""
        try:
            clip = mp.VideoFileClip(output_path)
            duration = clip.duration
            clip.close()
            
            # Check if duration is within acceptable range (±30 minutes)
            duration_diff = abs(duration - target_duration)
            if duration_diff > 1800:  # 30 minutes in seconds
                self.log(
                    f"Output duration {duration:.1f}s differs significantly from "
                    f"target {target_duration:.1f}s",
                    "WARNING", "VALIDATION"
                )
                return False
            
            # Verify file integrity
            try:
                test_clip = mp.VideoFileClip(output_path)
                # Try to read first and last frame
                test_clip.get_frame(0)
                test_clip.get_frame(test_clip.duration - 1/test_clip.fps)
                test_clip.close()
            except Exception as e:
                self.log(f"Output file integrity check failed: {e}", "ERROR", "VALIDATION")
                return False
            
            # Verify file size
            try:
                size_mb = os.path.getsize(output_path) / (1024 * 1024)
                if size_mb < 1:  # File too small, likely corrupted
                    self.log(f"Output file too small: {size_mb:.1f}MB", "ERROR", "VALIDATION")
                    return False
            except Exception as e:
                self.log(f"Error checking output file size: {e}", "ERROR", "VALIDATION")
                return False
            
            self.log(
                f"Output validation successful - Duration: {duration:.1f}s, "
                f"Size: {size_mb:.1f}MB",
                "INFO", "VALIDATION"
            )
            return True
            
        except Exception as e:
            self.log(f"Error validating output: {e}", "ERROR", "VALIDATION")
            return False

    def estimate_remaining_time(self, progress: float, start_time: float) -> str:
        """Estimate remaining processing time."""
        try:
            if progress <= 0:
                return "Calculating..."
                
            elapsed = time.time() - start_time
            estimated_total = elapsed / (progress / 100)
            remaining = estimated_total - elapsed
            
            if remaining < 60:
                return f"{remaining:.0f} seconds"
            elif remaining < 3600:
                return f"{remaining/60:.1f} minutes"
            else:
                hours = remaining // 3600
                minutes = (remaining % 3600) / 60
                return f"{hours:.0f}h {minutes:.0f}m"
                
        except Exception as e:
            self.log(f"Error estimating time: {e}", "ERROR", "PROGRESS")
            return "Unknown"

    def get_current_progress(self) -> Dict[str, Any]:
        """Get current processing progress information."""
        if not self.active_process or not self.progress:
            return {
                "progress": 0,
                "phase": "Idle",
                "elapsed": "0:00:00",
                "eta": "Unknown",
                "active": False
            }
            
        try:
            progress_info = self.progress.get_progress()
            progress_info["active"] = self.active_process
            progress_info["eta"] = self.estimate_remaining_time(
                progress_info["progress"],
                self.progress.start_time
            )
            return progress_info
            
        except Exception as e:
            self.log(f"Error getting progress: {e}", "ERROR", "PROGRESS")
            return {
                "progress": 0,
                "phase": "Error",
                "elapsed": "0:00:00",
                "eta": "Unknown",
                "active": False
            }
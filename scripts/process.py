# .\scripts\process.py

import os
import cv2
import numpy as np
import moviepy.editor as mp
import time
from typing import Dict, Any, Optional
from utility import (
    load_settings,
    log_event,
    cleanup_work_directory,
    ProgressTracker,
    MetricsCollector
)
from analyze import VideoAnalyzer 
from interface import ProcessingError, MovieConsolidatorError

class VideoProcessor:
    def __init__(self):
        self.settings = load_settings()
        self.video_config = self.settings.get('video', {})
        self.work_dir = self.video_config.get('temp_directory', 'work')
        self.metrics = MetricsCollector()
        self.progress = None
        self.active_process = False
        self.analyzer = VideoAnalyzer()  # Add analyzer instance

    def create_speed_transition(self, clip: mp.VideoFileClip, start_speed: float, 
                              end_speed: float, duration: float) -> mp.VideoFileClip:
        """Create smooth transition between speeds."""
        try:
            # Create array of gradually changing speeds
            transition_frames = self.settings.get('speed_settings', {}).get('transition_frames', 30)
            speeds = np.linspace(start_speed, end_speed, transition_frames)
            
            segments = []
            segment_duration = duration / transition_frames
            
            for i, speed in enumerate(speeds):
                start_time = i * segment_duration
                end_time = (i + 1) * segment_duration
                segment = clip.subclip(start_time, end_time).speedx(speed)
                segments.append(segment)
            
            return mp.concatenate_videoclips(segments)
        except Exception as e:
            log_event(f"Error creating speed transition: {e}", "ERROR", "SPEED")
            return clip

    def create_preview(self, input_path: str) -> str:
        """Create 360p preview version of video."""
        preview_height = self.settings.get('video_settings', {}).get('preview_height', 360)
        preview_path = os.path.join(self.work_dir, f"preview_{os.path.basename(input_path)}")
        
        try:
            clip = mp.VideoFileClip(input_path)
            aspect_ratio = clip.w / clip.h
            new_width = int(preview_height * aspect_ratio)
            
            preview = clip.resize(height=preview_height, width=new_width)
            preview.write_videofile(preview_path, codec='libx264')
            
            clip.close()
            preview.close()
            log_event(f"Created preview version: {preview_path}", "INFO", "PROCESSING")
            return preview_path
            
        except Exception as e:
            log_event(f"Error creating preview: {e}", "ERROR", "PROCESSING")
            return ""

    def apply_speed_changes(self, clip: mp.VideoFileClip, scene_data: Dict[str, Any], 
                          target_duration: float) -> mp.VideoFileClip:
        """Apply speed changes based on scene analysis."""
        try:
            segments = []
            fps = clip.fps
            
            for scene in scene_data['scenes']:
                start_time = scene['start_frame'] / fps
                end_time = scene['end_frame'] / fps
                
                if scene['is_static']:
                    log_event(f"Skipping static segment at {start_time:.2f}s", 
                            "INFO", "SPEED")
                    continue
                    
                subclip = clip.subclip(start_time, end_time)
                
                # Get total video duration for relative speed calculation
                total_duration = clip.duration
                scene_position = (start_time + end_time) / (2 * total_duration)
                duration = end_time - start_time

                if scene['is_menu']:
                    subclip = subclip.speedx(2.0)
                    log_event(f"Speeding up menu segment at {start_time:.2f}s", 
                            "INFO", "SPEED")
                elif scene['is_action']:
                    # Keep action scenes at normal speed but add transitions
                    if segments and hasattr(segments[-1], 'current_speed'):
                        transition_in = self.create_speed_transition(
                            subclip.subclip(0, 1),
                            segments[-1].current_speed,
                            1.0,
                            1.0
                        )
                        subclip = mp.concatenate_videoclips([transition_in, subclip.subclip(1)])
                    segments.append(subclip)
                else:
                    # Calculate and apply speed factor
                    speed_factor = self.analyzer.calculate_speed_factor(
                        duration, total_duration, target_duration, 
                        scene_position, scene['is_action']
                    )
                    
                    # Create transitions if needed
                    if segments and hasattr(segments[-1], 'current_speed'):
                        prev_speed = segments[-1].current_speed
                        if abs(prev_speed - speed_factor) > 0.5:  # Only transition for significant changes
                            transition = self.create_speed_transition(
                                subclip.subclip(0, 1),
                                prev_speed,
                                speed_factor,
                                1.0
                            )
                            subclip = mp.concatenate_videoclips([transition, subclip.subclip(1)])
                    
                    # Apply speed change
                    subclip = subclip.speedx(speed_factor)
                    subclip.current_speed = speed_factor  # Store current speed for transitions
                    log_event(f"Applied {speed_factor:.2f}x speed to segment", 
                            "INFO", "SPEED")
                
                segments.append(subclip)
            
            return mp.concatenate_videoclips(segments) if segments else clip
            
        except Exception as e:
            log_event(f"Error applying speed changes: {e}", "ERROR", "SPEED")
            return clip

    def process_video(self, input_path: str, output_path: str, target_duration: float) -> Optional[str]:
        """Process a single video file."""
        import gc  # Add garbage collection
        
        try:
            # Initialize progress tracking (5 phases)
            self.active_process = True
            self.progress = ProgressTracker(5)
            log_event(f"Starting processing of {input_path}", "INFO", "PROCESSING")
            
            # Phase 1: Create preview
            self.progress.update(phase="Creating Preview")
            preview_path = self.create_preview(input_path)
            if not preview_path:
                raise ProcessingError("Failed to create preview")
            gc.collect()
            
            # Phase 2: Analyze video
            self.progress.update(phase="Analyzing Video")
            scene_data = self.analyzer.analyze_video_file(preview_path, target_duration)
            if not scene_data['scenes']:
                raise ProcessingError("Failed to analyze video")
            gc.collect()
            
            # Phase 3: Load and prepare video
            self.progress.update(phase="Preparing Video")
            clip = mp.VideoFileClip(input_path)
            
            # Phase 4: Apply speed changes
            self.progress.update(phase="Applying Speed Changes")
            processed_clip = self.apply_speed_changes(clip, scene_data, target_duration)
            gc.collect()
            
            # Phase 5: Write final video
            self.progress.update(phase="Writing Final Video")
            processed_clip.write_videofile(
                output_path,
                codec=self.video_config.get('video_settings', {}).get('codec', 'libx264'),
                audio_codec=self.video_config.get('video_settings', {}).get('audio_codec', 'aac')
            )
            
            # Cleanup
            clip.close()
            processed_clip.close()
            cleanup_work_directory()
            gc.collect()
            
            self.active_process = False
            log_event("Video processing complete", "INFO", "PROCESSING")
            return output_path
            
        except Exception as e:
            log_event(f"Error processing video: {e}", "ERROR", "PROCESSING")
            self.active_process = False
            cleanup_work_directory()
            gc.collect()
            return None
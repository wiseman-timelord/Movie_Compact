# process.py

from dataclasses import dataclass
import cv2
import numpy as np
import moviepy.editor as mp
from typing import Dict, Any, Optional, List, Tuple, Callable, Union
from queue import Queue
from threading import Event, Lock
import time
import os
from utility import (
    load_settings,
    log_event,
    cleanup_work_directory,
    ProgressMonitor,
    MetricsCollector,
    PreviewGenerator,
    SceneManager,
    AudioAnalyzer,
    AudioProcessor,
    MemoryManager,
    ErrorHandler,
    CoreUtilities,
    detect_static_frame,
    detect_menu_screen,
    monitor_memory_usage
)
from scripts.temporary import (
    PROCESSING_CONFIG,
    AUDIO_CONFIG,
    SPEED_CONFIG,
    MEMORY_CONFIG,
    GLOBAL_STATE,
    SceneData,
    update_processing_state
)
from analyze import VideoAnalyzer
from interface import ProcessingError

class VideoProcessor:
    """Handles video processing and consolidation operations."""
    
    def __init__(self, log_manager=None):
        self.core = CoreUtilities()
        self.config = PROCESSING_CONFIG
        self.audio_config = AUDIO_CONFIG
        self.speed_config = SPEED_CONFIG
        self.memory_config = MEMORY_CONFIG
        
        self.analyzer = VideoAnalyzer(log_manager)
        self.audio_processor = AudioProcessor()
        self.memory_manager = MemoryManager()
        self.progress = ProgressMonitor()
        self.error_handler = ErrorHandler()
        self.metrics = MetricsCollector()
        self.log_manager = log_manager
        self.cancel_flag = Event()
        self._lock = Lock()

    def process_video(self, input_path: str, output_path: str, 
                     target_duration: float,
                     progress_callback: Optional[Callable] = None) -> bool:
        """Process a video file according to analysis results."""
        try:
            with self._lock:
                self.cancel_flag.clear()
                self.progress.start_stage("Processing")
                
                # Get scene data from global state if available
                scenes = GLOBAL_STATE.detected_scenes if GLOBAL_STATE.detected_scenes else []
                
                # Phase 1: Analysis if not already done
                if not scenes:
                    analysis = self._analyze_video(input_path, target_duration)
                    if not analysis or self.cancel_flag.is_set():
                        return False
                    scenes = [SceneData(**scene) for scene in analysis['scenes']]
                
                # Phase 2: Scene Processing
                processed_scenes = self._process_scenes(input_path, scenes)
                if not processed_scenes or self.cancel_flag.is_set():
                    return False
                
                # Phase 3: Final Compilation
                success = self._compile_video(
                    processed_scenes,
                    output_path,
                    GLOBAL_STATE.current_video.fps if GLOBAL_STATE.current_video else 30.0
                )
                
                self.progress.complete_stage("Processing")
                self.memory_manager.cleanup()
                
                return success
                
        except Exception as e:
            self.error_handler.handle_error(e, "video_processing")
            return False

    def _analyze_video(self, input_path: str, 
                      target_duration: float) -> Optional[Dict[str, Any]]:
        """Analyze video content."""
        self.progress.update_progress(0, "Analyzing video")
        try:
            analysis = self.analyzer.analyze_video(input_path, target_duration)
            if not analysis['scenes']:
                raise ProcessingError("No scenes detected in video")
            return analysis
        except Exception as e:
            self.error_handler.handle_error(e, "video_analysis")
            return None

    def _process_scenes(self, input_path: str, 
                       scenes: List[SceneData]) -> List[mp.VideoFileClip]:
        """Process individual scenes."""
        try:
            clip = mp.VideoFileClip(input_path)
            processed_scenes = []
            total_scenes = len(scenes)
            
            update_processing_state(
                stage="Processing",
                progress=0,
                processed_scenes=0,
                total_scenes=total_scenes
            )

            for i, scene in enumerate(scenes):
                if self.cancel_flag.is_set():
                    break

                progress = (i + 1) / total_scenes * 50
                self.progress.update_progress(
                    progress,
                    f"Processing scene {i+1}/{total_scenes}"
                )

                scene_clip = clip.subclip(
                    scene.start_frame / GLOBAL_STATE.current_video.fps,
                    scene.end_frame / GLOBAL_STATE.current_video.fps
                )

                processed_clip = self._process_scene_segment(
                    scene_clip,
                    scene
                )
                processed_scenes.append(processed_clip)

                update_processing_state(
                    processed_scenes=i+1,
                    progress=progress
                )

                if self.memory_manager.check_memory()['warning']:
                    self.memory_manager.cleanup()

            clip.close()
            return processed_scenes

        except Exception as e:
            self.error_handler.handle_error(e, "scene_processing")
            return []

    def _process_scene_segment(self, clip: mp.VideoFileClip,
                             scene: Dict[str, Any],
                             frame_rate: float) -> mp.VideoFileClip:
        """Process an individual scene segment."""
        try:
            # Handle static scenes
            if scene.get('is_static', False):
                return self._process_static_scene(clip)

            # Handle menu scenes
            if scene.get('is_menu', False):
                return self._process_menu_scene(clip)

            # Handle action scenes
            if scene.get('is_action', False):
                return clip.copy()

            # Apply dynamic speed adjustment
            processed = self._apply_speed_adjustment(
                clip,
                scene['speed_factor'],
                scene.get('transitions', [])
            )

            # Process audio
            if clip.audio is not None and self.config.enhance_audio:
                processed = self._process_audio(processed, scene['speed_factor'])

            return processed

        except Exception as e:
            self.error_handler.handle_error(e, "segment_processing")
            return clip.copy()

    def _process_static_scene(self, clip: mp.VideoFileClip) -> mp.VideoFileClip:
        """Process a static scene."""
        duration = min(1.0, clip.duration)
        return clip.subclip(0, duration)

    def _process_menu_scene(self, clip: mp.VideoFileClip) -> mp.VideoFileClip:
        """Process a menu scene."""
        return self._apply_speed_adjustment(clip, min(4.0, 2.0), [])

    def _apply_speed_adjustment(self, clip: mp.VideoFileClip,
                              speed_factor: float,
                              transitions: List[Dict[str, Any]]) -> mp.VideoFileClip:
        """Apply speed adjustments to clip."""
        try:
            if speed_factor == 1.0:
                return clip.copy()

            if clip.duration > 2.0:
                # Create smooth speed transition
                part1_duration = min(1.0, clip.duration * 0.2)
                part3_duration = min(1.0, clip.duration * 0.2)
                part2_duration = clip.duration - part1_duration - part3_duration

                # Process parts with transitions
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
                return clip.speedx(speed_factor)

        except Exception as e:
            self.error_handler.handle_error(e, "speed_adjustment")
            return clip.copy()

    def _create_speed_transition(self, clip: mp.VideoFileClip,
                               start_speed: float,
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
            self.error_handler.handle_error(e, "speed_transition")
            return clip.speedx(end_speed)

    def _create_smooth_transition(self, clip1: mp.VideoFileClip, 
                                clip2: mp.VideoFileClip,
                                transition_frames: int = 30) -> mp.VideoFileClip:
        """Create smooth transition between clips."""
        try:
            if clip1.duration < 1 or clip2.duration < 1:
                return mp.concatenate_videoclips([clip1, clip2])
                
            # Create transition frames
            transition_duration = transition_frames / clip1.fps
            fade_out = clip1.subclip(-transition_duration)
            fade_in = clip2.subclip(0, transition_duration)
            
            # Create blended frames
            frames = []
            for i in range(transition_frames):
                progress = i / transition_frames
                frame1 = fade_out.get_frame(i/clip1.fps)
                frame2 = fade_in.get_frame(i/clip2.fps)
                blended = frame1 * (1-progress) + frame2 * progress
                frames.append(blended)
                
            transition = mp.ImageSequenceClip(frames, fps=clip1.fps)
            
            # Combine clips with transition
            final = mp.concatenate_videoclips([
                clip1.subclip(0, -transition_duration),
                transition,
                clip2.subclip(transition_duration)
            ])
            
            return final


    def _process_audio(self, clip: mp.VideoFileClip,
                      speed_factor: float) -> mp.VideoFileClip:
        """Process audio for the clip."""
        try:
            if clip.audio is None:
                return clip

            # Extract audio data
            audio_array = clip.audio.to_soundarray()
            
            # Process audio
            processed_audio = self.audio_processor.process_audio(
                audio_array,
                speed_factor
            )

            # Create new audio clip
            audio_clip = mp.AudioArrayClip(processed_audio, clip.audio.fps)
            
            # Set processed audio
            return clip.set_audio(audio_clip)

        except Exception as e:
            self.error_handler.handle_error(e, "audio_processing")
            return clip

    def _compile_video(self, scenes: List[mp.VideoFileClip],
                      output_path: str,
                      fps: float) -> bool:
        """Compile processed scenes into final video."""
        try:
            self.progress.update_progress(80, "Compiling final video")

            # Create output directory if needed
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Concatenate scenes
            final_clip = mp.concatenate_videoclips(scenes)

            # Write final video
            final_clip.write_videofile(
                output_path,
                codec=self.config.output_codec,
                audio_codec=self.config.audio_codec,
                fps=fps,
                threads=self.config.threads
            )

            final_clip.close()
            return True

        except Exception as e:
            self.error_handler.handle_error(e, "video_compilation")
            return False

    def cancel_processing(self) -> None:
        """Cancel current processing operation."""
        self.cancel_flag.set()
        log_event("Processing cancelled by user", "INFO", "CONTROL")

    def validate_output(self, output_path: str, target_duration: float) -> bool:
        """Validate the processed video."""
        try:
            clip = mp.VideoFileClip(output_path)
            duration = clip.duration
            clip.close()

            # Check duration
            duration_diff = abs(duration - target_duration)
            if duration_diff > 1800:  # 30 minutes
                log_event(
                    f"Output duration {duration:.1f}s differs significantly from "
                    f"target {target_duration:.1f}s",
                    "WARNING",
                    "VALIDATION"
                )
                return False

            # Verify file integrity
            test_clip = mp.VideoFileClip(output_path)
            test_clip.get_frame(0)
            test_clip.get_frame(test_clip.duration - 1/test_clip.fps)
            test_clip.close()

            # Verify file size
            size_mb = os.path.getsize(output_path) / (1024 * 1024)
            if size_mb < 1:
                log_event(f"Output file too small: {size_mb:.1f}MB",
                         "ERROR", "VALIDATION")
                return False

            return True

        except Exception as e:
            self.error_handler.handle_error(e, "output_validation")
            return False

    def _create_smooth_transition(self, clip1: mp.VideoFileClip, 
                                clip2: mp.VideoFileClip,
                                transition_frames: int = 30) -> mp.VideoFileClip:
        """Create smooth transition between clips."""
        try:
            if clip1.duration < 1 or clip2.duration < 1:
                return mp.concatenate_videoclips([clip1, clip2])
                
            # Create transition frames
            transition_duration = transition_frames / clip1.fps
            fade_out = clip1.subclip(-transition_duration)
            fade_in = clip2.subclip(0, transition_duration)
            
            # Create blended frames
            frames = []
            for i in range(transition_frames):
                progress = i / transition_frames
                frame1 = fade_out.get_frame(i/clip1.fps)
                frame2 = fade_in.get_frame(i/clip2.fps)
                blended = frame1 * (1-progress) + frame2 * progress
                frames.append(blended)
                
            transition = mp.ImageSequenceClip(frames, fps=clip1.fps)
            
            # Combine clips with transition
            final = mp.concatenate_videoclips([
                clip1.subclip(0, -transition_duration),
                transition,
                clip2.subclip(transition_duration)
            ])
            
            return final
            
        except Exception as e:
            self.error_handler.handle_error(e, "transition_creation")
            return mp.concatenate_videoclips([clip1, clip2])

class BatchProcessor:
    """Handle batch processing of multiple videos."""
    
    def __init__(self):
        self.processor = VideoProcessor()
        self.queue = Queue()
        self.active = False
        self._lock = Lock()

    def add_to_queue(self, input_path: str, output_path: str,
                    target_duration: float) -> None:
        """Add a video to the processing queue."""
        self.queue.put((input_path, output_path, target_duration))

    def process_queue(self, progress_callback: Optional[Callable] = None) -> None:
        """Process all videos in the queue."""
        with self._lock:
            self.active = True
            
        while not self.queue.empty() and self.active:
            input_path, output_path, target_duration = self.queue.get()
            try:
                self.processor.process_video(
                    input_path,
                    output_path,
                    target_duration,
                    progress_callback
                )
            except Exception as e:
                log_event(f"Error processing {input_path}: {e}",
                         "ERROR", "BATCH")
            finally:
                self.queue.task_done()

        self.active = False

def _process_parallel(self, files: List[Tuple[str, str, float]]) -> None:
        """Process multiple files in parallel."""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import threading
        
        self.processing_lock = threading.Lock()
        self.completed_files = 0
        self.total_files = len(files)
        
        def process_file(input_path: str, output_path: str, 
                        target_duration: float) -> bool:
            try:
                result = self.processor.process_video(
                    input_path,
                    output_path,
                    target_duration,
                    progress_callback=self._update_batch_progress
                )
                
                with self.processing_lock:
                    self.completed_files += 1
                    self.queue_status[input_path]['status'] = (
                        'Complete' if result else 'Failed'
                    )
                return result
                
            except Exception as e:
                self.error_handler.handle_error(e, "parallel_processing")
                return False
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            # Submit all files for processing
            future_to_file = {
                executor.submit(
                    process_file,
                    input_path,
                    output_path,
                    target_duration
                ): input_path
                for input_path, output_path, target_duration in files
            }
            
            # Monitor completion
            for future in as_completed(future_to_file):
                input_path = future_to_file[future]
                try:
                    success = future.result()
                    if not success:
                        self.log_manager.log(
                            f"Failed to process {input_path}",
                            "ERROR",
                            "BATCH"
                        )
                except Exception as e:
                    self.error_handler.handle_error(e, "batch_monitoring")

    def get_queue_status(self) -> List[Dict[str, Any]]:
        """Get current status of processing queue."""
        status_list = []
        for file_path, info in self.queue_status.items():
            status_list.append({
                'name': os.path.basename(file_path),
                'status': info['status'],
                'progress': info['progress']
            })
        return status_list

    def _update_batch_progress(self, file_path: str, progress: float) -> None:
        """Update progress for a specific file in the batch."""
        with self.processing_lock:
            self.queue_status[file_path]['progress'] = progress
            
            # Calculate overall progress
            total_progress = sum(
                info['progress'] for info in self.queue_status.values()
            )
            overall_progress = total_progress / len(self.queue_status)
            
            if self.progress_callback:
                self.progress_callback(
                    'Batch Processing',
                    overall_progress,
                    f"Completed {self.completed_files}/{self.total_files} files"
                )

    def cancel_processing(self) -> None:
        """Cancel batch processing."""
        self.active = False
        self.processor.cancel_processing()
        while not self.queue.empty():
            self.queue.get()
            self.queue.task_done()
            

# Script: `.\scripts\process.py`

# Imports
import cv2, time, os, librosa
from dataclasses import dataclass
import numpy as np
import pyopencl as cl
import moviepy.editor as mp
from typing import Dict, Any, Optional, List, Tuple, Callable, Union
from queue import Queue
from threading import Event, Lock
from scripts.analyze import VideoAnalyzer
from scripts.exceptions import ProcessingError
from concurrent.futures import ThreadPoolExecutor, as_completed
from scripts.temporary import (
    PROCESSING_CONFIG,
    AUDIO_CONFIG,
    SPEED_CONFIG,
    MEMORY_CONFIG,
    GLOBAL_STATE,
    SceneData,
    update_processing_state,
    BASE_DIR
)
from scripts.utility import (
    load_settings,
    cleanup_work_directory,
    ProgressMonitor,
    MetricsCollector,
    PreviewGenerator,
    SceneManager,
    AudioAnalyzer,
    MemoryManager,
    ErrorHandler,
    CoreUtilities,
    detect_static_frame,
    detect_menu_screen,
    AudioProcessor
)

# SceneProcessor class
class SceneProcessor:
    """Handles dynamic scene processing based on scene type."""
    def __init__(self, config: Dict[str, Any], audio_processor: AudioProcessor):
        self.config = config
        self.audio_processor = audio_processor
        self.hardware_ctx = hardware_ctx
        self.sample_rate = sample_rate

    def process(self, scene_type: str, clip: mp.VideoFileClip, scene: SceneData) -> mp.VideoFileClip:
        """Process a scene based on its type."""
        if scene_type == 'static':
            return self._process_static(clip)
        elif scene_type == 'menu':
            return self._process_menu(clip)
        elif scene_type == 'action':
            return clip.copy()
        else:
            processed = self._apply_speed_adjustment(clip, scene.speed_factor, scene.transitions)
            if clip.audio is not None and self.config['enhance_audio']:
                processed = self._process_audio(processed, scene.speed_factor)
            return processed

    def _process_static(self, clip: mp.VideoFileClip) -> mp.VideoFileClip:
        """Process a static scene."""
        duration = min(1.0, clip.duration)
        return clip.subclip(0, duration)

    def _process_menu(self, clip: mp.VideoFileClip) -> mp.VideoFileClip:
        """Process a menu scene."""
        return self._apply_speed_adjustment(clip, min(4.0, 2.0), [])

    def _apply_speed_adjustment(self, clip: mp.VideoFileClip, speed_factor: float, 
                                transitions: List[Dict[str, Any]]) -> mp.VideoFileClip:
        """Apply speed adjustments with transitions."""
        try:
            if speed_factor == 1.0:
                return clip.copy()
            if clip.duration > 2.0:
                part1_duration = min(1.0, clip.duration * 0.2)
                part3_duration = min(1.0, clip.duration * 0.2)
                part2_duration = clip.duration - part1_duration - part3_duration
                part1 = self._create_speed_transition(clip.subclip(0, part1_duration), 1.0, speed_factor)
                part2 = clip.subclip(part1_duration, part1_duration + part2_duration).speedx(speed_factor)
                part3 = self._create_speed_transition(clip.subclip(clip.duration - part3_duration), speed_factor, 1.0)
                return mp.concatenate_videoclips([part1, part2, part3])
            return clip.speedx(speed_factor)
        except Exception as e:
            print(f"Error in speed adjustment: {e}")
            return clip.copy()

    def _process_audio(self, clip: mp.VideoFileClip, speed_factor: float) -> mp.VideoFileClip:
        """
        Process audio for a video clip, optimized for AAC from AVC/H.264.
        """
        if clip.audio is None:
            return clip
        audio_processor = AudioProcessor(sample_rate=44100)  # Common AAC sample rate
        audio_array = clip.audio.to_soundarray()
        # Handle AAC audio specifically
        if clip.audio.fps == 44100 and clip.audio.nchannels == 2:  # Typical AAC characteristics
            processed_audio = audio_processor.process_audio(audio_array, speed_factor)
        else:
            processed_audio = audio_array
        return clip.set_audio(mp.AudioArrayClip(processed_audio, fps=clip.audio.fps))

    def _create_speed_transition(self, clip: mp.VideoFileClip, start_speed: float, 
                                 end_speed: float) -> mp.VideoFileClip:
        """Create smooth transition between speeds."""
        try:
            n_frames = int(clip.duration * clip.fps)
            if n_frames < 2:
                return clip.speedx(end_speed)
            speeds = np.linspace(start_speed, end_speed, n_frames)
            frames = [clip.get_frame(i / clip.fps) for i, speed in enumerate(speeds)]
            return mp.ImageSequenceClip(frames, fps=clip.fps)
        except Exception as e:
            print(f"Error in speed transition: {e}")
            return clip.speedx(end_speed)

# VideoProcessor class
class VideoProcessor:
    """Optimized video processing engine."""
    def __init__(self, settings=None, analyzer=None, hardware_ctx):
        self.cancel_flag = Event()
        self.settings = settings or load_settings()
        self.core = CoreUtilities()
        self.config = PROCESSING_CONFIG
        self.audio_config = AUDIO_CONFIG
        self.speed_config = SPEED_CONFIG
        self.memory_config = MEMORY_CONFIG
        self.hardware_capabilities = self.settings['hardware_config']
        
        self.analyzer = analyzer  # Use provided analyzer
        self.audio_processor = AudioProcessor()
        self.memory_manager = MemoryManager(hardware_ctx)  # Add this line
        self.progress = ProgressMonitor()
        self.error_handler = ErrorHandler()
        self.metrics = MetricsCollector()
        self.sample_rate = self.audio_config.get('sample_rate', 44100)
        self.scene_processor = SceneProcessor(self.config, self.audio_processor, self.hardware_ctx, self.sample_rate)
        self.hardware_ctx = hardware_ctx
        self.kernel = None
        self._lock = Lock()
        if self.hardware_ctx.get('use_opencl', False):
            self._setup_opencl()

    def cancel_processing(self) -> None:
        self.cancel_flag.set()

    @profile  # Memory profiling decorator
    def process_video(self, input_path, output_path, scenes, target_duration):
        audio_files = []
        frame_count = 0

        chunk_size = 1024**3  # 1GB chunks
        for chunk_frames in self.memory_manager.stream_video(input_path, chunk_size):
            if self.cancel_flag.is_set():
                break
            # Process each chunk (placeholder for actual processing logic)
            processed_chunk = self._process_chunk(chunk_frames)
            # Accumulate or write processed chunk to output (implement as needed)
            if progress_callback:
                progress_callback("Processing chunk...")
        # Finalize output (implement as needed)
        if progress_callback:
            progress_callback("Video processing complete")

        # Stream video frames
        memory_manager = MemoryManager()
        for frames in memory_manager.stream_video(input_path):
            if frames.size == 0:
                continue
            # Simulate OpenCL processing (e.g., frame differencing)
            if self.hardware_ctx['use_opencl']:
                mf = cl.mem_flags
                frame1_buf = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=frames[0].flatten())
                frame2_buf = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=frames[1].flatten() if len(frames) > 1 else frames[0].flatten())
                diff_buf = cl.Buffer(self.context, mf.WRITE_ONLY, frames[0].size)
                self.program.frame_diff(self.queue, frames[0].shape[:2], None, frame1_buf, frame2_buf, diff_buf, np.int32(frames.shape[2]), np.int32(frames.shape[1]))
                diff = np.empty_like(frames[0].flatten())
                cl.enqueue_copy(self.queue, diff, diff_buf)
            else:
                # AVX2 or NumPy fallback (assuming AOCL if available)
                diff = np.abs(frames[0] - frames[1]) if len(frames) > 1 else frames[0]
            frame_count += len(frames)

        # Process audio for each scene
        for i, scene in enumerate(scenes):
            start_time = scene['start_time']
            end_time = scene['end_time']
            speed_factor = scene['speed_factor']
            temp_audio = f"temp_audio_{i}.wav"
            filter_chain = f"rubberband=tempo={speed_factor}" if self.has_rubberband() else f"atempo={speed_factor}"
            (
                ffmpeg
                .input(input_path, ss=start_time, to=end_time)
                .filter_(filter_chain)
                .output(temp_audio)
                .run(quiet=True)
            )
            audio_files.append(temp_audio)

        # Concatenate audio files
        audio_list_file = "audio_list.txt"
        with open(audio_list_file, 'w') as f:
            for audio_file in audio_files:
                f.write(f"file '{audio_file}'\n")
        concatenated_audio = "concatenated_audio.wav"
        (
            ffmpeg
            .input(audio_list_file, format='concat', safe=0)
            .output(concatenated_audio)
            .run(quiet=True)
        )

        # Simulate writing output (simplified; in practice, use moviepy or ffmpeg to merge)
        probe = ffmpeg.probe(input_path)
        duration = float(probe['format']['duration'])
        print(f"Processed {frame_count} frames, original duration: {duration}s, target duration: {target_duration}s")

        # Clean up
        for audio_file in audio_files:
            os.remove(audio_file)
        os.remove(audio_list_file)
        os.remove(concatenated_audio)

    def _setup_opencl(self):
        """Set up OpenCL kernel for frame differencing optimized for RX470."""
        if not self.hardware_ctx.get('use_opencl', False) or not self.hardware_ctx.get('context'):
            return
        context = self.hardware_ctx['context']
        queue = cl.CommandQueue(context)
        program = cl.Program(context, """
            __kernel void frame_diff(__global const uchar *frame1, __global const uchar *frame2,
                                   __global uchar *diff, int width, int height) {
                int x = get_global_id(0);
                int y = get_global_id(1);
                if (x < width && y < height) {
                    int idx = (y * width + x) * 3;
                    diff[idx] = abs(frame1[idx] - frame2[idx]);
                    diff[idx + 1] = abs(frame1[idx + 1] - frame2[idx + 1]);
                    diff[idx + 2] = abs(frame1[idx + 2] - frame2[idx + 2]);
                }
            }
        """).build()
        self.kernel = program.frame_diff
        self.queue = queue

    def has_rubberband(self):
        try:
            result = subprocess.run(['ffmpeg', '-filters'], capture_output=True, text=True)
            return 'rubberband' in result.stdout
        except:
            return False

    def _process_chunk(self, chunk_frames):
        """
        Process a chunk of frames using OpenCL for frame differencing and motion detection.
        """
        if self.kernel and len(chunk_frames) >= 2:
            frame1 = cl.Buffer(self.hardware_ctx['context'], cl.mem_flags.READ_ONLY, size=chunk_frames[0].nbytes)
            frame2 = cl.Buffer(self.hardware_ctx['context'], cl.mem_flags.READ_ONLY, size=chunk_frames[1].nbytes)
            diff = cl.Buffer(self.hardware_ctx['context'], cl.mem_flags.WRITE_ONLY, size=chunk_frames[0].nbytes)
            cl.enqueue_copy(self.queue, frame1, chunk_frames[0].tobytes())
            cl.enqueue_copy(self.queue, frame2, chunk_frames[1].tobytes())
            self.kernel(self.queue, chunk_frames[0].shape[:2], (16, 16), frame1, frame2, diff,
                        np.int32(chunk_frames[0].shape[1]), np.int32(chunk_frames[0].shape[0]))
            result = np.empty_like(chunk_frames[0])
            cl.enqueue_copy(self.queue, result, diff)
            
            # Motion detection example
            motion_score = np.mean(result)
            if motion_score > self.config.get('processing_config', {}).get('motion_threshold', 0.3):
                # Placeholder: Process as motion scene (e.g., adjust speed)
                return chunk_frames  # Return original frames for now
            else:
                # Placeholder: Skip or compress static scene
                return [chunk_frames[0]]  # Return first frame as static
        return chunk_frames  # Fallback to CPU

    def process_video(self, input_path, output_path, target_duration, progress_callback=None):
        """
        Optimized video processing with chunking and OpenCL support.
        """
        chunk_size = 1024**3  # 1GB chunks
        for chunk_frames in self.memory_manager.stream_video(input_path, chunk_size):
            if self.cancel_flag.is_set():
                break
            processed_chunk = self._process_chunk(chunk_frames)
            # Write or accumulate processed chunk (implement as needed)
            if progress_callback:
                progress_callback("Processing chunk...")
        if progress_callback:
            progress_callback("Video processing complete")

    def process_audio(self, input_path, speed_factor=1.0):
            """Process audio from AVC/MP4 files, preserving characteristics."""
            try:
                audio, sr = librosa.load(input_path, sr=None)
                if speed_factor != 1.0:
                    audio = librosa.effects.time_stretch(audio, rate=speed_factor)
                return audio, sr
            except Exception as e:
                print(f"Error processing audio: {e}")
                return np.array([]), 0

    def _analyze_video(self, input_path: str, 
                      target_duration: float) -> Optional[Dict[str, Any]]:
        """Analyze video content."""
        self.progress.update_progress(0, "Analyzing video")
        try:
            if self.analyzer is None:
                raise ProcessingError("VideoAnalyzer not provided")
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

            # Preload video properties to avoid redundant access
            fps = clip.fps
            if GLOBAL_STATE.current_video is None or GLOBAL_STATE.current_video.fps != fps:
                GLOBAL_STATE.current_video = type('Video', (), {'fps': fps})()

            for i, scene in enumerate(scenes):
                if self.cancel_flag.is_set():
                    print("Info: Processing canceled by user")
                    break

                progress = 40 + ((i + 1) / total_scenes * 50)  # Start at 40%, end at 90%
                self.progress.update_progress(
                    progress,
                    f"Processing scene {i+1}/{total_scenes}"
                )

                # Calculate scene time boundaries
                start_time = scene.start_frame / fps
                end_time = scene.end_frame / fps
                
                # Ensure valid time range
                if start_time >= clip.duration or end_time <= start_time:
                    print(f"Warning: Skipping invalid scene {i+1}: {start_time:.2f}s - {end_time:.2f}s")
                    continue

                scene_clip = clip.subclip(start_time, min(end_time, clip.duration))

                # Process the scene segment with proper audio handling
                processed_clip = self.scene_processor.process(scene.scene_type, scene_clip, scene)
                if processed_clip:
                    processed_scenes.append(processed_clip)
                else:
                    print(f"Warning: Scene {i+1} processing failed, skipping")

                update_processing_state(
                    processed_scenes=len(processed_scenes),
                    progress=progress
                )

                # Memory management with detailed feedback
                memory_status = self.memory_manager.check_memory()
                if memory_status['warning']:
                    print("Info: Memory warning detected, triggering cleanup")
                    self.memory_manager.cleanup()
                    if self.memory_manager.check_memory()['critical']:
                        print("Error: Critical memory shortage after cleanup, aborting")
                        break

            clip.close()
            self.progress.update_progress(90, "Scene processing complete")
            return processed_scenes

        except Exception as e:
            self.error_handler.handle_error(e, "scene_processing")
            print(f"Error: Scene processing failed - {str(e)}")
            return []
        finally:
            if 'clip' in locals():
                clip.close()

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
        print("Info: Processing cancelled by user")
        time.sleep(1)

    def validate_output(self, output_path: str, target_duration: float) -> bool:
        """Validate the processed video."""
        try:
                from scripts.utility import extract_frames_optimized
                processed_scenes = []
                total_scenes = len(scenes)
                frame_gen = extract_frames_optimized(input_path, batch_size=self.config['performance']['frame_buffer_size'])
                frames = list(frame_gen)  # Collect frames efficiently
                fps = GLOBAL_STATE.current_video.fps if GLOBAL_STATE.current_video else 30.0

                update_processing_state(stage="Processing", progress=0, processed_scenes=0, total_scenes=total_scenes)

                for i, scene in enumerate(scenes):
                    if self.cancel_flag.is_set():
                        break
                    progress = 40 + ((i + 1) / total_scenes * 50)
                    self.progress.update_progress(progress, f"Processing scene {i+1}/{total_scenes}")

                    start_idx = scene.start_frame
                    end_idx = min(scene.end_frame, len(frames) - 1)
                    if start_idx >= len(frames) or end_idx <= start_idx:
                        continue

                    # Create clip from frame subset
                    scene_frames = frames[start_idx:end_idx + 1]
                    scene_clip = mp.ImageSequenceClip(scene_frames, fps=fps)
                    processed_clip = self.scene_processor.process(scene.scene_type, scene_clip, scene)
                    processed_scenes.append(processed_clip)

                    update_processing_state(processed_scenes=len(processed_scenes), progress=progress)

                    if self.memory_manager.check_memory()['warning']:
                        self.memory_manager.cleanup()

                self.progress.update_progress(90, "Scene processing complete")
                return processed_scenes
            except Exception as e:
                self.error_handler.handle_error(e, "scene_processing")
                return []

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
    def __init__(self, settings=None, analyzer=None):
        self.processor = VideoProcessor(settings=settings, analyzer=analyzer)
        self.error_handler = ErrorHandler()
        self.queue = Queue()
        self.queue_status = {}
        self.progress_callback = None
        self.active = True
        self.processing_lock = Lock()

    def add_to_queue(self, input_path: str, output_path: str, target_duration: float) -> None:
        """Add a file to the processing queue."""
        print(f"Info: Added {input_path} to queue")
        self.queue.put((input_path, output_path, target_duration))
        self.queue_status[input_path] = {'status': 'Pending', 'progress': 0.0}
        time.sleep(1)

    def process_queue(self, progress_callback: Optional[Callable] = None) -> None:
        """Process all files in the queue."""
        self.progress_callback = progress_callback
        files = []
        while not self.queue.empty() and self.active:
            files.append(self.queue.get())
            self.queue.task_done()
        if files:
            self._process_parallel(files)
        print("Info: Queue processing complete")
        time.sleep(1)

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
            
            total_progress = sum(info['progress'] for info in self.queue_status.values())
            overall_progress = total_progress / len(self.queue_status) if self.queue_status else 0
            
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
        print("Info: Batch processing cancelled")
        time.sleep(1)

    def _process_parallel(self, files: List[Tuple[str, str, float]]) -> None:
        """Process multiple files in parallel."""
        import threading
        
        self.completed_files = 0
        self.total_files = len(files)
        
        def process_file(input_path: str, output_path: str, target_duration: float) -> bool:
            try:
                result = self.processor.process_video(
                    input_path,
                    output_path,
                    target_duration,
                    progress_callback=lambda p, msg: self._update_batch_progress(input_path, p)
                )
                
                with self.processing_lock:
                    self.completed_files += 1
                    self.queue_status[input_path]['status'] = 'Complete' if result else 'Failed'
                return result
                
            except Exception as e:
                self.error_handler.handle_error(e, "parallel_processing")
                with self.processing_lock:
                    self.queue_status[input_path]['status'] = 'Failed'
                return False
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_file = {
                executor.submit(
                    process_file,
                    input_path,
                    output_path,
                    target_duration
                ): input_path
                for input_path, output_path, target_duration in files
            }
            
            for future in as_completed(future_to_file):
                input_path = future_to_file[future]
                try:
                    success = future.result()
                    if not success:
                        print(f"Error: Failed to process {input_path}")
                        time.sleep(5)
                except Exception as e:
                    self.error_handler.handle_error(e, "batch_monitoring")
                    print(f"Error: Exception in batch processing {input_path}: {e}")

class PreviewGenerator:
    def __init__(self, work_dir: Optional[str] = None):
        self.work_dir = work_dir if work_dir is not None else get_full_path('work')
        os.makedirs(self.work_dir, exist_ok=True)

    def create_preview(self, input_path: str) -> str:
        """
        Create a 360p preview of the video using ffmpeg.

        Args:
            input_path: Path to the input video.

        Returns:
            str: Path to the preview video if successful, otherwise an empty string.
        """
        preview_path = os.path.join(self.work_dir, f"preview_{os.path.basename(input_path)}")
        command = f'ffmpeg -i "{input_path}" -vf "scale=-1:360" -c:v libx264 -crf 23 -c:a aac -y "{preview_path}"'

        try:
            result = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if result.returncode == 0:
                return preview_path
            else:
                print(f"ffmpeg failed with return code {result.returncode}")
                return ""
        except subprocess.CalledProcessError as e:
            print(f"Error creating preview: {e}")
            return ""

    def generate_thumbnails(self, video_path: str, num_thumbnails: int = 5) -> List[str]:
        """Generate thumbnails from video for preview."""
        try:
            import moviepy.editor as mp
            clip = mp.VideoFileClip(video_path)
            duration = clip.duration
            interval = duration / (num_thumbnails + 1)
            
            thumbnails = []
            for i in range(num_thumbnails):
                time = interval * (i + 1)
                frame = clip.get_frame(time)
                
                # Save thumbnail
                thumb_path = os.path.join(
                    self.work_dir,
                    f"thumb_{i}_{os.path.basename(video_path)}.jpg"
                )
                cv2.imwrite(thumb_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                thumbnails.append(thumb_path)
            
            clip.close()
            return thumbnails
            
        except Exception as e:
            print(f"Error: Thumbnail generation failed - {e}")
            time.sleep(5)

class SceneManager:
    def __init__(self, scene_config):
        self.scene_settings = scene_config
        self.min_scene_length = scene_config['min_scene_length']
        self.max_scene_length = scene_config['max_scene_length']
        self.threshold = scene_config['scene_threshold']
        self._lock = Lock()
        self.metrics = MetricsCollector()

    def detect_scenes(self, frames: List[np.ndarray],
                     audio_segments: List[Tuple[float, float]]) -> List[Dict[str, Any]]:
        """Detect and analyze scenes combining visual and audio information."""
        with self._lock:
            scenes = []
            current_scene = None
            
            for i in range(1, len(frames)):
                if self.is_scene_change(frames[i-1], frames[i]):
                    if current_scene:
                        scenes.append(self.finalize_scene(current_scene, frames, i-1))
                    current_scene = self.initialize_scene(i)
                
                if current_scene:
                    current_scene = self.update_scene(current_scene, frames[i], i)
                
                if (current_scene and 
                    i - current_scene['start_frame'] > self.max_scene_length * 30):
                    scenes.append(self.finalize_scene(current_scene, frames, i))
                    current_scene = self.initialize_scene(i + 1)
            
            if current_scene:
                scenes.append(self.finalize_scene(current_scene, frames, len(frames)-1))
            
            scenes = self.merge_short_scenes(scenes)
            scenes = self.merge_audio_info(scenes, audio_segments, len(frames))
            
            return scenes

    def is_scene_change(self, frame1: np.ndarray, frame2: np.ndarray) -> bool:
        """Detect if there is a scene change between frames."""
        try:
            hist1 = cv2.calcHist([frame1], [0, 1, 2], None, [8, 8, 8],
                               [0, 256, 0, 256, 0, 256])
            hist2 = cv2.calcHist([frame2], [0, 1, 2], None, [8, 8, 8],
                               [0, 256, 0, 256, 0, 256])
            diff = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CHISQR)
            
            frame_diff = cv2.absdiff(frame1, frame2)
            diff_score = np.mean(frame_diff)
            
            return diff > self.threshold or diff_score > 30.0
        except Exception as e:
            print(f"Error: Scene change detection failed - {e}")
            time.sleep(5)

    def initialize_scene(self, start_frame: int) -> Dict[str, Any]:
        return {
            'start_frame': start_frame,
            'end_frame': None,
            'is_action': False,
            'is_menu': False,
            'is_static': False,
            'motion_score': 0.0,
            'audio_activity': 0.0,
            'frame_count': 0,
            'complexity': 0.0,
            'transitions': [],
            'prev_frame': None  # Add prev_frame
        }

    def _process_scene_segment(self, clip: mp.VideoFileClip, scene: SceneData) -> mp.VideoFileClip:
        try:
            if scene.scene_type == 'static':
                return self._process_static_scene(clip)
            if scene.scene_type == 'menu':
                return self._process_menu_scene(clip)
            if scene.scene_type == 'action':
                return clip.copy()

            # Apply speed adjustment to video only
            processed_video = self._apply_speed_adjustment(clip, scene.speed_factor, scene.transitions)

            # Process audio if present
            if clip.audio is not None and self.config['enhance_audio']:
                original_audio = clip.audio.to_soundarray()
                new_duration = processed_video.duration
                original_duration = clip.duration
                rate = original_duration / new_duration if new_duration > 0 else 1.0

                processed_audio = self.audio_processor.process_audio(original_audio, rate)
                audio_clip = mp.AudioArrayClip(processed_audio, fps=clip.audio.fps)
                processed = processed_video.set_audio(audio_clip)
            else:
                processed = processed_video

            return processed
        except Exception as e:
            self.error_handler.handle_error(e, "segment_processing")
            print(f"Error: Failed to process scene segment - {e}")
            time.sleep(5)
            return clip.copy()

    def _calculate_scene_complexity(self, frame: np.ndarray, current_complexity: float,
                                  frame_count: int) -> float:
        """Calculate scene complexity based on various metrics."""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Calculate edge density
            edges = cv2.Canny(gray, 100, 200)
            edge_density = np.count_nonzero(edges) / edges.size
            
            # Calculate texture complexity
            texture_score = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Calculate color variety
            color_score = np.std(frame.reshape(-1, 3), axis=0).mean()
            
            # Combine metrics
            complexity = (0.4 * edge_density + 
                        0.3 * (texture_score / 1000) +  # Normalize texture score
                        0.3 * (color_score / 255))      # Normalize color score
            
            # Update running average
            if frame_count == 1:
                return complexity
            else:
                return (current_complexity * (frame_count - 1) + complexity) / frame_count
                
        except Exception as e:
            print(f"Error: Scene complexity calculation failed - {e}")
            time.sleep(5)


    def finalize_scene(self, scene: Dict[str, Any], frames: List[np.ndarray],
                      end_frame: int) -> Dict[str, Any]:
        """Finalize scene data and analysis."""
        scene['end_frame'] = end_frame
        scene['is_static'] = all(
            detect_static_frame(frames[i], frames[i+1])
            for i in range(scene['start_frame'], end_frame-1)
        )
        
        # Calculate final metrics
        scene['duration'] = (end_frame - scene['start_frame']) / 30  # Assuming 30fps
        scene['average_motion'] = scene['motion_score']
        
        # Detect transitions
        scene['transitions'] = self._detect_transitions(
            frames[scene['start_frame']:end_frame+1]
        )
        
        return scene

    def _detect_transitions(self, frames: List[np.ndarray]) -> List[Dict[str, Any]]:
        """Detect scene transitions between frames."""
        transitions = []
        for i in range(1, len(frames) - 1):
            try:
                # Check frame consistency
                if frames[i].shape != frames[i-1].shape or frames[i].shape != frames[i+1].shape:
                    print(f"Warning: Frame shape mismatch at index {i}")
                    time.sleep(3)
                    continue

                diff_prev = cv2.absdiff(frames[i], frames[i-1])
                diff_next = cv2.absdiff(frames[i], frames[i+1])
                mean_diff_prev = np.mean(diff_prev)
                mean_diff_next = np.mean(diff_next)

                if mean_diff_prev < 5 and mean_diff_next > 30:
                    transitions.append({'frame': i, 'type': 'cut', 'confidence': 0.9})
                elif mean_diff_prev < mean_diff_next:
                    transitions.append({'frame': i, 'type': 'fade', 'confidence': 0.7})
            except Exception as e:  # Single except block for all exceptions
                print(f"Warning: Transition detection error - {e}")
                time.sleep(3)
        return transitions

    def merge_short_scenes(self, scenes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge scenes that are too short."""
        if not scenes:
            return scenes
            
        merged = []
        current = scenes[0]
        
        for next_scene in scenes[1:]:
            if (next_scene['start_frame'] - current['end_frame'] < 
                self.min_scene_length * 30):
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
                current['complexity'] = (current['complexity'] + next_scene['complexity']) / 2
                current['transitions'].extend(next_scene['transitions'])
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
            
            # Update scene importance based on audio
            scene['importance_score'] = self._calculate_importance(scene)
        
        return scenes

    def _calculate_importance(self, scene: Dict[str, Any]) -> float:
        """Calculate scene importance based on various factors."""
        weights = {
            'motion': 0.3,
            'audio': 0.3,
            'complexity': 0.2,
            'duration': 0.2
        }
        
        # Normalize duration score (prefer scenes between 2-10 seconds)
        duration_score = 1.0 - abs(scene['duration'] - 6) / 6
        duration_score = max(0, min(1, duration_score))
        
        scores = {
            'motion': scene['motion_score'],
            'audio': scene['audio_activity'],
            'complexity': scene['complexity'],
            'duration': duration_score
        }
        
        return sum(weights[k] * scores[k] for k in weights)
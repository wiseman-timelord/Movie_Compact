# interface.py

import os
import json
import gradio as gr
import sys
from typing import Dict, Any, Optional, Tuple, List
from utility import (
    load_hardware_config,
    load_settings,
    log_event,
    LogManager,
    MetricsCollector,
    FileProcessor
)
from process import VideoProcessor
from threading import Lock, Thread
import time
from queue import Queue
import datetime

class MovieCompactError(Exception):
    """Base exception for Movie Consolidator errors."""
    pass

class ProcessingError(MovieCompactError):
    """Raised when video processing fails."""
    pass

class AnalysisError(MovieCompactError):
    """Raised when video analysis fails."""
    pass

class HardwareError(MovieCompactError):
    """Raised when hardware-related operations fail."""
    pass

class ConfigurationError(MovieCompactError):
    """Raised when configuration-related operations fail."""
    pass

class LogUpdateManager:
    """Manages real-time log updates for the UI."""
    def __init__(self, log_file: str):
        self.log_file = log_file
        self.last_position = 0
        self.lock = Lock()
        self.active = False
        self.update_thread = None
        self.update_queue = Queue()
        self.callbacks = []

    def add_callback(self, callback):
        """Add a callback function to be called when logs update."""
        self.callbacks.append(callback)

    def remove_callback(self, callback):
        """Remove a callback function."""
        if callback in self.callbacks:
            self.callbacks.remove(callback)

    def start_monitoring(self):
        """Start monitoring log file for changes."""
        self.active = True
        self.update_thread = Thread(target=self._monitor_log)
        self.update_thread.daemon = True
        self.update_thread.start()

    def stop_monitoring(self):
        """Stop monitoring log file."""
        self.active = False
        if self.update_thread:
            self.update_thread.join(timeout=1.0)

    def _monitor_log(self):
        """Monitor log file for changes and trigger updates."""
        while self.active:
            try:
                with self.lock:
                    if os.path.exists(self.log_file):
                        with open(self.log_file, 'r') as f:
                            f.seek(0, 2)  # Seek to end
                            if f.tell() < self.last_position:
                                self.last_position = 0  # File was truncated
                            f.seek(self.last_position)
                            new_content = f.read()
                            if new_content:
                                self.last_position = f.tell()
                                self.update_queue.put(new_content)
                                for callback in self.callbacks:
                                    try:
                                        callback(new_content)
                                    except Exception as e:
                                        print(f"Error in log update callback: {e}")
            except Exception as e:
                print(f"Error monitoring log: {e}")
                time.sleep(1)  # Prevent rapid retries on error
            time.sleep(0.1)  # Prevent excessive CPU usage

class GradioInterface:
    def __init__(self, log_manager: Optional[LogManager] = None):
        self.settings = load_settings()
        self.hardware_config = load_hardware_config()
        self.video_config = self.settings.get('video', {})
        self.metrics = MetricsCollector()
        self.processor = VideoProcessor()
        self.active_process = False
        self.file_processor = FileProcessor(
            self.video_config.get('supported_formats', ['.mp4', '.avi', '.mkv'])
        )
        self.log_manager = log_manager or LogManager(os.path.join("data", "events.txt"))
        self.log_updater = LogUpdateManager(os.path.join("data", "events.txt"))
        self.processing_lock = Lock()
        self.current_video_info = None

    def load_persistent_settings(self) -> Dict[str, Any]:
        """Load settings from persistent.json."""
        try:
            persistent_file = os.path.join("data", "persistent.json")
            if os.path.exists(persistent_file):
                with open(persistent_file, "r") as f:
                    return json.load(f)
            return {}
        except Exception as e:
            self.log_manager.log(f"Error loading settings: {e}", "ERROR", "CONFIG")
            return {}

    def save_persistent_settings(self, settings: Dict[str, Any]) -> None:
        """Save settings to persistent.json."""
        try:
            persistent_file = os.path.join("data", "persistent.json")
            with open(persistent_file, "w") as f:
                json.dump(settings, f, indent=4)
            self.log_manager.log("Settings saved successfully", "INFO", "CONFIG")
        except Exception as e:
            self.log_manager.log(f"Error saving settings: {e}", "ERROR", "CONFIG")

    def update_event_log(self) -> str:
        """Get recent logs for display."""
        try:
            event_file = os.path.join("data", "events.txt")
            if not os.path.exists(event_file):
                return "No events logged yet."
                
            with open(event_file, "r") as f:
                lines = f.readlines()
                
            # Get last 50 lines for more context
            recent_logs = lines[-50:]
            formatted_logs = []
            for line in recent_logs:
                # Parse timestamp and format nicely if possible
                try:
                    parts = line.split("[", 2)
                    timestamp = parts[1].split("]")[0]
                    dt = datetime.datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
                    formatted_time = dt.strftime("%H:%M:%S")
                    message = parts[2].strip()
                    formatted_logs.append(f"{formatted_time} | {message}")
                except:
                    formatted_logs.append(line.strip())
            
            return "\n".join(formatted_logs)
        except Exception as e:
            return f"Error reading event log: {e}"

    def update_metrics_display(self) -> str:
        """Get current metrics for display."""
        if not self.active_process:
            return "No active processing."
        return self.metrics.get_metrics_report()

    def get_video_info(self, video_path: str) -> Tuple[str, float]:
        """Get information about the selected video."""
        try:
            import moviepy.editor as mp
            clip = mp.VideoFileClip(video_path)
            duration = clip.duration
            size_mb = os.path.getsize(video_path) / (1024 * 1024)
            fps = clip.fps
            resolution = f"{clip.w}x{clip.h}"
            
            # Store video info for later use
            self.current_video_info = {
                'path': video_path,
                'duration': duration,
                'size': size_mb,
                'fps': fps,
                'resolution': resolution,
                'has_audio': clip.audio is not None
            }
            
            # Get audio information
            audio_info = "No audio"
            if clip.audio is not None:
                audio_info = f"Audio: {clip.audio.fps}Hz"
            
            # Calculate estimated processing time
            est_time = self._estimate_processing_time(duration, size_mb)
            
            # Hardware acceleration info
            hw_info = "Hardware Acceleration: "
            if self.hardware_config.get("OpenCL"):
                hw_info += "OpenCL (GPU)"
            elif self.hardware_config.get("Avx2"):
                hw_info += "AVX2 (CPU)"
            else:
                hw_info += "Standard CPU"
            
            clip.close()

            info = (
                f"Video Information:\n"
                f"Duration: {duration:.1f} seconds ({duration/60:.1f} minutes)\n"
                f"Size: {size_mb:.1f} MB\n"
                f"FPS: {fps}\n"
                f"Resolution: {resolution}\n"
                f"{audio_info}\n"
                f"{hw_info}\n"
                f"Estimated Processing Time: {est_time}"
            )
            return info, duration
        except Exception as e:
            self.log_manager.log(f"Error getting video info: {e}", "ERROR", "INFO")
            return "Error reading video file", 0.0

    def _estimate_processing_time(self, duration: float, size_mb: float) -> str:
        """Estimate processing time based on video properties."""
        # Base time calculation
        base_minutes = (duration / 60) * 0.5  # Rough estimate: 30% of video duration
        
        # Adjust for file size
        if size_mb > 1000:  # For files over 1GB
            base_minutes *= 1.5
            
        # Adjust for hardware
        if self.hardware_config.get("OpenCL"):
            base_minutes *= 0.7  # 30% faster with GPU
        elif not self.hardware_config.get("Avx2"):
            base_minutes *= 1.3  # 30% slower without AVX2
            
        # Add buffer for safety
        estimated_minutes = base_minutes * 1.2
        
        if estimated_minutes < 60:
            return f"~{estimated_minutes:.0f} minutes"
        else:
            hours = estimated_minutes // 60
            mins = estimated_minutes % 60
            return f"~{hours:.0f}h {mins:.0f}m"

    def process_video(self, video_path: str, target_minutes: float,
                     progress: gr.Progress = gr.Progress()) -> str:
        """Process a single video file with progress updates."""
        if self.active_process:
            return "Processing already in progress. Please wait."
            
        with self.processing_lock:
            try:
                self.active_process = True
                if not video_path:
                    return "No video file selected."

                # Validate input
                if not os.path.exists(video_path):
                    return "Selected file does not exist."
                if target_minutes <= 0:
                    return "Target duration must be greater than 0."

                # Validate target duration is reasonable
                if self.current_video_info:
                    current_duration = self.current_video_info['duration'] / 60
                    if target_minutes >= current_duration:
                        return (f"Target duration ({target_minutes:.1f} min) must be less than "
                               f"original duration ({current_duration:.1f} min)")
                    if target_minutes < current_duration * 0.1:
                        return (f"Target duration ({target_minutes:.1f} min) is too short. "
                               f"Minimum recommended: {(current_duration * 0.1):.1f} min")

                output_dir = self.settings.get("paths", {}).get("output_path", "output")
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                    
                # Generate unique output filename
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = os.path.join(
                    output_dir, 
                    f"processed_{timestamp}_{os.path.basename(video_path)}"
                )

                self.log_manager.log(
                    f"Starting video processing: {os.path.basename(video_path)}", 
                    "INFO", 
                    "PROCESSING"
                )
                self.metrics.start_phase_timing("processing")

                # Convert minutes to seconds for processing
                target_duration = target_minutes * 60
                
                # Process the video with progress updates
                result = self.processor.process_video(
                    video_path, 
                    output_path, 
                    target_duration,
                    progress_callback=self._update_progress
                )
                
                self.metrics.end_phase_timing("processing")
                
                if result:
                    if self.processor.validate_output(output_path, target_duration):
                        # Get final video info
                        final_info = self.get_video_info(output_path)[0]
                        
                        success_msg = (
                            f"Processing complete!\n\n"
                            f"Output saved to: {output_path}\n\n"
                            f"Final Video Info:\n{final_info}\n\n"
                            f"Processing Report:\n{self.metrics.get_metrics_report()}"
                        )
                        return success_msg
                    else:
                        return (
                            f"Processing completed but output validation failed.\n"
                            f"Please check the output file: {output_path}\n\n"
                            f"Processing Report:\n{self.metrics.get_metrics_report()}"
                        )
                else:
                    return "Processing failed. Check the logs for details."
                
            except Exception as e:
                error_msg = f"Processing failed: {e}"
                self.log_manager.log(error_msg, "ERROR", "PROCESSING")
                return error_msg
            finally:
                self.active_process = False

    def _update_progress(self, phase: str, progress: float, message: str) -> None:
        """Update progress information and logs."""
        self.log_manager.log(f"[{phase}] {progress:.1f}% - {message}", "INFO", "PROGRESS")

    def launch_interface(self):
        """Launch the Gradio interface with real-time updates."""
        settings = self.load_persistent_settings()
        
        with gr.Blocks(title="Movie Consolidator", theme=gr.themes.Base()) as interface:
            gr.Markdown(
                """
                # Movie Consolidator
                Convert long gaming sessions into concise highlight reels.
                Upload a video and specify your target duration in minutes.
                """
            )
            
            with gr.Tabs():
                # Main Processing Tab
                with gr.Tab("Process Video"):
                    with gr.Row():
                        with gr.Column(scale=2):
                            # Input Section
                            with gr.Group():
                                gr.Markdown("### Select Input Video")
                                video_input = gr.File(
                                    label="Select Video File",
                                    file_types=[".mp4", ".avi", ".mkv"],
                                    interactive=True
                                )
                                video_info = gr.TextArea(
                                    label="Video Information",
                                    interactive=False,
                                    lines=8,
                                    show_copy_button=True
                                )
                                target_duration = gr.Number(
                                    label="Target Duration (minutes)",
                                    value=30,
                                    minimum=1,
                                    interactive=True,
                                    info="Desired length of the final video in minutes"
                                )
                            
                            # Process Section
                            with gr.Group():
                                gr.Markdown("### Processing Controls")
                                with gr.Row():
                                    process_button = gr.Button(
                                        "Start Processing", 
                                        variant="primary",
                                        interactive=True
                                    )
                                    cancel_button = gr.Button(
                                        "Cancel Processing",
                                        variant="secondary",
                                        interactive=False
                                    )
                                    
                                current_phase = gr.Textbox(
                                    label="Current Processing Phase",
                                    value="Idle",
                                    interactive=False
                                )
                                progress_bar = gr.Progress(
                                    label="Processing Progress",
                                    show_progress=True,
                                    interactive=False
                                )
                                status_output = gr.TextArea(
                                    label="Processing Status",
                                    value="Ready for processing...",
                                    interactive=False,
                                    lines=6,
                                    show_copy_button=True
                                )

                        # Monitoring Column
                        with gr.Column(scale=1):
                            # Event Log
                            gr.Markdown("### Live Processing Log")
                            event_log = gr.TextArea(
                                label="Recent Events",
                                value="Waiting for processing to begin...",
                                interactive=False,
                                lines=15,
                                autoscroll=True,
                                show_copy_button=True
                            )
                            
                            # Metrics Display
                            gr.Markdown("### Processing Metrics")
                            metrics_display = gr.TextArea(
                                label="Current Metrics",
                                value="No processing metrics available.",
                                interactive=False,
                                lines=10,
                                show_copy_button=True
                            )
                            
                            with gr.Row():
                                refresh_log = gr.Button(
                                    "Refresh Log", 
                                    variant="secondary"
                                )
                                clear_log = gr.Button(
                                    "Clear Log",
                                    variant="secondary"
                                )

                # Settings Tab
                with gr.Tab("Settings"):
                    with gr.Row():
                        # Video Settings
                        with gr.Column():
                            gr.Markdown("### Video Settings")
                            preview_res = gr.Slider(
                                label="Preview Resolution",
                                minimum=240,
                                maximum=720,
                                step=120,
                                value=settings.get("video", {}).get("preview_height", 360),
                                info="Height of preview video (in pixels)"
                            )
                            output_res = gr.Slider(
                                label="Output Resolution",
                                minimum=480,
                                maximum=1080,
                                step=120,
                                value=settings.get("video", {}).get("output_height", 720),
                                info="Height of output video (in pixels)"
                            )
                            output_quality = gr.Slider(
                                label="Output Quality",
                                minimum=1,
                                maximum=5,
                                step=1,
                                value=settings.get("video", {}).get("quality", 3),
                                info="1 = Highest compression, 5 = Best quality"
                            )

                        # Processing Settings
                        with gr.Column():
                            gr.Markdown("### Processing Settings")
                            motion_threshold = gr.Slider(
                                label="Motion Detection Sensitivity",
                                minimum=0.1,
                                maximum=1.0,
                                step=0.1,
                                value=settings.get("processing", {}).get("motion_threshold", 0.3),
                                info="Lower values detect more subtle movements"
                            )
                            scene_threshold = gr.Slider(
                                label="Scene Change Sensitivity",
                                minimum=0.1,
                                maximum=1.0,
                                step=0.1,
                                value=settings.get("processing", {}).get("scene_threshold", 0.5),
                                info="Lower values detect more scene changes"
                            )
                            min_scene_duration = gr.Number(
                                label="Minimum Scene Duration (seconds)",
                                value=settings.get("processing", {}).get("min_scene_duration", 2.0),
                                minimum=0.5,
                                info="Minimum duration to consider as a scene"
                            )

                    with gr.Row():
                        # Speed Settings
                        with gr.Column():
                            gr.Markdown("### Speed Settings")
                            max_speed = gr.Slider(
                                label="Maximum Speed Factor",
                                minimum=1.5,
                                maximum=8.0,
                                step=0.5,
                                value=settings.get("speed", {}).get("max_speed", 4.0),
                                info="Maximum speed multiplier for non-action scenes"
                            )
                            action_threshold = gr.Slider(
                                label="Action Scene Threshold",
                                minimum=0.1,
                                maximum=1.0,
                                step=0.1,
                                value=settings.get("processing", {}).get("action_threshold", 0.6),
                                info="Threshold for detecting action scenes"
                            )

                        # Audio Settings
                        with gr.Column():
                            gr.Markdown("### Audio Settings")
                            audio_threshold = gr.Slider(
                                label="Audio Activity Threshold",
                                minimum=0.1,
                                maximum=1.0,
                                step=0.1,
                                value=settings.get("audio", {}).get("threshold", 0.4),
                                info="Threshold for detecting significant audio"
                            )
                            preserve_pitch = gr.Checkbox(
                                label="Preserve Audio Pitch",
                                value=settings.get("audio", {}).get("preserve_pitch", True),
                                info="Maintain audio pitch during speed changes"
                            )

                    with gr.Row():
                        save_settings = gr.Button(
                            "Save Settings",
                            variant="primary"
                        )
                        reset_settings = gr.Button(
                            "Reset to Defaults",
                            variant="secondary"
                        )

                # Help Tab
                with gr.Tab("Help"):
                    gr.Markdown("""
                    # Movie Consolidator Help
                    
                    ## Overview
                    Movie Consolidator helps you convert long gaming sessions into concise, 
                    action-packed videos by:
                    - Detecting and removing static screens and menus
                    - Identifying and preserving action sequences
                    - Intelligently adjusting playback speed
                    - Maintaining video quality while reducing duration
                    
                    ## Usage Guide
                    
                    ### 1. Select Input Video
                    - Click "Select Video File" to choose your input video
                    - Supported formats: MP4, AVI, MKV
                    - Check the video information display for file details
                    
                    ### 2. Set Target Duration
                    - Enter your desired output length in minutes
                    - The program will try to reach this target while preserving important content
                    - Recommended: Target between 20-40% of original length
                    
                    ### 3. Process Video
                    - Click "Start Processing" to begin
                    - Monitor progress in the live processing log
                    - Processing time depends on video length and settings
                    
                    ### 4. Settings (Optional)
                    - Video Settings: Adjust quality and resolution
                    - Processing Settings: Fine-tune detection sensitivity
                    - Speed Settings: Control playback speed changes
                    - Audio Settings: Configure audio processing
                    
                    ## Tips for Best Results
                    1. Use good quality source videos
                    2. Avoid videos with complex transitions
                    3. Start with default settings
                    4. Monitor the processing log
                    5. Adjust settings if needed
                    
                    ## Troubleshooting
                    - If processing fails, check the log for details
                    - Ensure enough disk space is available
                    - Try reducing output quality for large files
                    - Clear the log if it becomes too long
                    
                    ## Support
                    For additional help or to report issues, refer to:
                    - Project documentation
                    - GitHub repository
                    - Issue tracker
                    """)

            # Event Handlers
            def update_button_states(active: bool):
                """Update button states based on processing status."""
                return {
                    process_button: gr.update(interactive=not active),
                    cancel_button: gr.update(interactive=active),
                    video_input: gr.update(interactive=not active),
                    target_duration: gr.update(interactive=not active)
                }

            def clear_log_content():
                """Clear the log content."""
                try:
                    log_file = os.path.join("data", "events.txt")
                    with open(log_file, 'w') as f:
                        f.write("Log cleared\n")
                    return "Log cleared."
                except Exception as e:
                    return f"Error clearing log: {e}"

            # Connect event handlers
            video_input.change(
                fn=lambda x: self.get_video_info(x.name) if x else ("No file selected", 0),
                inputs=[video_input],
                outputs=[video_info]
            )

            process_button.click(
                fn=self.process_video,
                inputs=[video_input, target_duration],
                outputs=[status_output],
                _js="() => {  document.querySelector('#event-log').scrollTop = document.querySelector('#event-log').scrollHeight; }"
            ).then(
                fn=lambda: update_button_states(True),
                outputs=[process_button, cancel_button, video_input, target_duration]
            )

            cancel_button.click(
                fn=lambda: self.processor.cancel_processing() if hasattr(self.processor, 'cancel_processing') else None
            ).then(
                fn=lambda: update_button_states(False),
                outputs=[process_button, cancel_button, video_input, target_duration]
            )

            refresh_log.click(
                fn=lambda: (
                    self.update_event_log(),
                    self.update_metrics_display()
                ),
                outputs=[event_log, metrics_display]
            )

            clear_log.click(
                fn=clear_log_content,
                outputs=[event_log]
            )

            # Settings event handlers
            def save_current_settings(
                preview_res, output_res, output_quality,
                motion_threshold, scene_threshold, min_scene_duration,
                max_speed, action_threshold,
                audio_threshold, preserve_pitch
            ):
                """Save current settings to persistent storage."""
                settings = {
                    "video": {
                        "preview_height": preview_res,
                        "output_height": output_res,
                        "quality": output_quality
                    },
                    "processing": {
                        "motion_threshold": motion_threshold,
                        "scene_threshold": scene_threshold,
                        "min_scene_duration": min_scene_duration,
                        "action_threshold": action_threshold
                    },
                    "speed": {
                        "max_speed": max_speed
                    },
                    "audio": {
                        "threshold": audio_threshold,
                        "preserve_pitch": preserve_pitch
                    }
                }
                self.save_persistent_settings(settings)
                return "Settings saved successfully!"

            def reset_default_settings():
                """Reset settings to default values."""
                default_settings = {
                    "preview_res": 360,
                    "output_res": 720,
                    "output_quality": 3,
                    "motion_threshold": 0.3,
                    "scene_threshold": 0.5,
                    "min_scene_duration": 2.0,
                    "max_speed": 4.0,
                    "action_threshold": 0.6,
                    "audio_threshold": 0.4,
                    "preserve_pitch": True
                }
                return [default_settings[key] for key in [
                    "preview_res", "output_res", "output_quality",
                    "motion_threshold", "scene_threshold", "min_scene_duration",
                    "max_speed", "action_threshold",
                    "audio_threshold", "preserve_pitch"
                ]]

            save_settings.click(
                fn=save_current_settings,
                inputs=[
                    preview_res, output_res, output_quality,
                    motion_threshold, scene_threshold, min_scene_duration,
                    max_speed, action_threshold,
                    audio_threshold, preserve_pitch
                ],
                outputs=[status_output]
            )

            reset_settings.click(
                fn=reset_default_settings,
                outputs=[
                    preview_res, output_res, output_quality,
                    motion_threshold, scene_threshold, min_scene_duration,
                    max_speed, action_threshold,
                    audio_threshold, preserve_pitch
                ]
            )

            # Start log monitoring when interface loads
            interface.load(self.log_updater.start_monitoring)
            
            # Launch interface
            interface.queue().launch(
                server_name="0.0.0.0",
                server_port=7860,
                share=False,
                inbrowser=True,
                show_error=True
            )  # Added closing parenthesis here

def launch_gradio_interface(log_manager: Optional[LogManager] = None):
    """Launch the Gradio interface."""
    try:
        interface = GradioInterface(log_manager)
        interface.launch_interface()
    except Exception as e:
        log_event(f"Failed to launch interface: {e}", "ERROR", "INTERFACE")
        print(f"Error: Failed to launch interface - {e}")
        sys.exit(1)

if __name__ == "__main__":
    launch_gradio_interface()
# interface.py

import os, json, time, sys, traceback, webbrowser
from threading import Timer
import gradio as gr
from typing import Dict, Any, Optional, Tuple, List, Callable
from datetime import datetime
from dataclasses import dataclass
from queue import Queue
from threading import Lock, Event
from scripts.exceptions import MovieCompactError, ProcessingError
from scripts.utility import (
    load_settings,
    MetricsCollector,
    FileProcessor,
    MemoryManager,
    ErrorHandler,
    CoreUtilities
)
from scripts.temporary import (
    PROCESSING_CONFIG,
    PROGRESS_CONFIG,
    ERROR_CONFIG,
    GLOBAL_STATE,
    ProcessingState,
    VideoMetadata,
    get_full_path,
    update_processing_state,
    AUDIO_CONFIG  # Add this line
)
from scripts.analyze import VideoAnalyzer
from scripts.process import VideoProcessor, BatchProcessor

# Classes...
class InterfaceManager:
    def __init__(self):
        self.core = CoreUtilities()  # Core utilities instance
        self.config = PROCESSING_CONFIG  # MUST COME FIRST
        self.file_processor = FileProcessor(
            supported_formats=self.config['supported_formats']
        )
        self.settings = load_settings()
        self.hardware_capabilities = self.settings['hardware_config']
        self.analyzer = VideoAnalyzer(settings=self.settings)
        self.processor = VideoProcessor(settings=self.settings, analyzer=self.analyzer)
        self.batch_processor = BatchProcessor()
        self.processing_lock = Lock()
        self.processor.progress.register_callback(self._update_progress)

    def _handle_batch_processing(self, selected_files: List[str], target_duration: float) -> str:
        """Handle batch processing of multiple video files."""
        if not selected_files:
            return "No files selected"
        with self.processing_lock:  # Ensure thread-safe processing
            for file in selected_files:
                input_path = os.path.join("input", file)
                output_path = os.path.join("output", f"processed_{int(time.time())}_{file}")
                self.batch_processor.add_to_queue(input_path, output_path, target_duration * 60)
            self.batch_processor.process_queue(self._update_progress)
            return "Batch processing complete"

    def _update_progress(self, stage: str, progress: float, message: str) -> None:
        """Update progress bar and status in a thread-safe manner."""
        with self.processing_lock:
            self.progress_bar.update(progress / 100)  # Assume progress_bar exists (e.g., GUI component)
            self.status_output.update(f"Stage: {stage}\nProgress: {progress:.1f}%\n{message}")  # Assume status_output exists

    def create_interface(self) -> gr.Blocks:
        """Create and configure the Gradio interface."""
        with gr.Blocks(title="Movie Consolidator", theme=gr.themes.Base()) as interface:
            self._create_header()
            
            with gr.Tabs():
                self._create_main_tab()
                self._create_batch_tab()
                self._create_settings_tab()
                self._create_help_tab()
            
            self._setup_event_handlers(interface)
            
            return interface

    def _create_header(self) -> None:
        """Create interface header."""
        gr.Markdown("""
        # Movie Consolidator
        Convert long videos into concise, action-packed highlight reels.
        """)

    def _create_main_tab(self) -> None:
        """Create main processing tab."""
        with gr.Tab("Process Video"):
            with gr.Row():
                with gr.Column(scale=2):
                    # Input Section
                    with gr.Group():
                        gr.Markdown("### Input Video")
                        self.video_input = gr.File(
                            label="Select Video",
                            file_types=[".mp4", ".avi", ".mkv"]
                        )
                        self.video_info = gr.TextArea(
                            label="Video Information",
                            interactive=False,
                            lines=8
                        )
                        self.target_duration = gr.Number(
                            label="Target Duration (minutes)",
                            value=30,
                            minimum=1
                        )

                    # Process Section
                    with gr.Group():
                        gr.Markdown("### Processing")
                        with gr.Row():
                            self.process_btn = gr.Button(
                                "Start Processing",
                                variant="primary"
                            )
                            self.cancel_btn = gr.Button(
                                "Cancel",
                                variant="secondary",
                                interactive=False
                            )

                        self.progress_bar = gr.Progress()
                        self.status_output = gr.TextArea(
                            label="Status",
                            value="Ready",
                            lines=6
                        )

                with gr.Column(scale=1):
                    # Monitoring Section
                    gr.Markdown("### Processing Log")
                    self.event_log = gr.TextArea(
                        label="Events",
                        value="Waiting for processing...",
                        lines=15,
                        autoscroll=True
                    )

                    gr.Markdown("### Metrics")
                    self.metrics_display = gr.TextArea(
                        label="Processing Metrics",
                        value="No metrics available",
                        lines=10
                    )

                    with gr.Row():
                        self.refresh_log_btn = gr.Button("Refresh Log")
                        self.clear_log_btn = gr.Button("Clear Log")

    def _create_batch_tab(self) -> None:
        with gr.Tab("Batch Processing"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Available Files")
                    files = self._get_input_files()
                    self.file_list = gr.CheckboxGroup(
                        label="Select Files to Process",
                        choices=files,
                        value=[],
                        interactive=bool(files)
                    )
                    self.batch_status_msg = gr.Markdown(
                        "⚠️ Add videos to input directory first",
                        visible=not bool(files)
                    )
                    
                    # Add refresh button
                    self.refresh_files_btn = gr.Button("Refresh File List")
                    
                    self.batch_target_duration = gr.Number(
                        label="Target Duration (minutes)",
                        value=30,
                        minimum=1
                    )
                    
                    with gr.Row():
                        self.batch_process_btn = gr.Button(
                            "Process Selected",
                            variant="primary",
                            interactive=False
                        )
                        self.batch_cancel_btn = gr.Button(
                            "Cancel Batch",
                            variant="secondary"
                        )
                
                with gr.Column():
                    self.batch_status = gr.TextArea(
                        label="Batch Status",
                        value="No batch processing active",
                        lines=15
                    )
                    self.batch_progress = gr.Progress()
                    self._create_batch_queue_display()  # Add queue display

    def _create_settings_tab(self) -> None:
        with gr.Tab("Settings"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Video Settings")
                    self.motion_threshold = gr.Slider(
                        minimum=0.1,
                        maximum=1.0,
                        value=ConfigManager.get('processing', 'scene_detection.motion_threshold'),
                        label="Motion Detection Sensitivity"
                    )
                    self.min_scene_duration = gr.Number(
                        value=ConfigManager.get('processing', 'scene_detection.min_scene_duration'),
                        label="Minimum Scene Duration (seconds)"
                    )
                    self.max_speed = gr.Slider(
                        minimum=1.0,
                        maximum=8.0,
                        value=ConfigManager.get('speed', 'max_speed_factor'),
                        label="Maximum Speed Factor"
                    )

                # Audio Settings
                with gr.Column():
                    gr.Markdown("### Audio Settings")
                    self.preserve_pitch = gr.Checkbox(
                        value=AUDIO_CONFIG['preserve_pitch'],  # Fixed line
                        label="Preserve Audio Pitch"
                    )
                    self.enhance_audio = gr.Checkbox(
                        value=AUDIO_CONFIG['enhance_audio'],   # Fixed line
                        label="Enhance Audio Quality"
                    )

            # Hardware Preferences
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Hardware Preferences")
                    # Get actual detected capabilities from hardware.json
                    hw_caps = self.settings['hardware_config']
                    
                    # Use explicit boolean checks with fallbacks
                    self.use_opencl = gr.Checkbox(
                        value=bool(self.config.get("hardware_acceleration", {}).get("use_opencl", False)),
                        label=f"Use OpenCL ({'Available' if hw_caps.get('OpenCL', False) else 'Unavailable'})",
                        interactive=hw_caps.get('OpenCL', False)
                    )
                    
                    self.use_avx2 = gr.Checkbox(
                        value=bool(self.config.get("hardware_preferences", {}).get("use_avx2", False)),
                        label=f"Use AVX2 ({'Available' if hw_caps.get('AVX2', False) else 'Unavailable'})",
                        interactive=hw_caps.get('AVX2', False)
                    )
                    
                    self.use_aocl = gr.Checkbox(
                        value=bool(self.config.get("hardware_preferences", {}).get("use_aocl", False)),
                        label=f"Use AOCL ({'available' if self.hardware_capabilities.get('AOCL', False) else 'not available'})",
                        interactive=bool(self.hardware_capabilities.get('AOCL', False))  # Explicit bool conversion
                    )

            with gr.Row():
                self.save_settings_btn = gr.Button(
                    "Save Settings",
                    variant="primary"
                )
                self.reset_settings_btn = gr.Button(
                    "Reset to Defaults",
                    variant="secondary"
                )

    def _create_help_tab(self) -> None:
        """Create help and documentation tab."""
        with gr.Tab("Help"):
            gr.Markdown("""
            # Movie Consolidator Help
            
            ## Overview
            Movie Consolidator helps you convert long videos into concise, 
            action-packed highlights by:
            - Detecting and removing static screens and menus
            - Identifying and preserving action sequences
            - Intelligently adjusting playback speed
            - Maintaining video quality while reducing duration
            
            ## Usage Guide
            
            ### Single Video Processing
            1. Select your input video file
            2. Set desired target duration
            3. Click "Start Processing"
            4. Monitor progress in the log
            
            ### Batch Processing
            1. Place videos in the input folder
            2. Select videos to process
            3. Set common target duration
            4. Click "Process Selected"
            
            ### Settings
            - Motion Detection: Adjust sensitivity to motion
            - Scene Duration: Set minimum scene length
            - Speed Factor: Maximum speed for non-action scenes
            - Audio Settings: Configure audio processing
            
            ## Tips
            - Use good quality source videos
            - Start with default settings
            - Monitor the processing log
            - Adjust settings if needed
            
            ## Troubleshooting
            - Check the log for errors
            - Ensure enough disk space
            - Try reducing quality for large files
            - Clear the log if it becomes too long
            """)

    def _setup_event_handlers(self, interface: gr.Blocks) -> None:
        """Setup event handlers for interface elements."""
        # File selection handlers
        self.refresh_files_btn.click(
                fn=self._update_file_list,
                inputs=None,
                outputs=[self.file_list, self.batch_status_msg, self.batch_process_btn]
            )

        # Processing handlers
        self.process_btn.click(
            fn=self._handle_processing,
            inputs=[self.video_input, self.target_duration],
            outputs=[self.status_output]
        ).then(
            fn=self._update_button_states,
            inputs=None,
            outputs=[self.process_btn, self.cancel_btn, self.video_input, self.target_duration]
        )

        self.cancel_btn.click(
            fn=self._handle_cancellation,
            inputs=None,
            outputs=[self.status_output]
        ).then(
            fn=self._update_button_states,
            inputs=None,
            outputs=[self.process_btn, self.cancel_btn, self.video_input, self.target_duration]
        )

        # Batch processing handlers
        self.batch_process_btn.click(
            fn=self._handle_batch_processing,
            inputs=[self.file_list, self.batch_target_duration],
            outputs=[self.batch_status],
            preprocess=False  # Disable Gradio's input validation
        )

        self.batch_cancel_btn.click(
            fn=self._handle_batch_cancellation,
            inputs=None,
            outputs=[self.batch_status],
            queue=False
        )

        # Settings handlers
        self.save_settings_btn.click(
                fn=self._save_settings,
                inputs=[
                    self.motion_threshold,
                    self.min_scene_duration,
                    self.max_speed,
                    self.enhance_audio,
                    self.use_opencl,
                    self.use_avx2,
                    self.use_aocl
                ],
                outputs=[self.status_output]
            )

        self.reset_settings_btn.click(
            fn=self._reset_settings,
            outputs=[
                self.motion_threshold,
                self.min_scene_duration,
                self.max_speed,
                self.preserve_pitch,
                self.enhance_audio,
                self.use_opencl,
                self.use_avx2,
                self.use_aocl
            ]
        )

        # Log handlers
        self.refresh_log_btn.click(
            fn=self._refresh_log,
            outputs=[self.event_log, self.metrics_display]
        )

        self.clear_log_btn.click(
            fn=self._clear_log,
            outputs=[self.event_log]
        )

    def _handle_file_selection(self, video_file: gr.File) -> str:
        """Handle video file selection."""
        if not video_file:
            return "No file selected"
            
        try:
            metadata = VideoMetadata(
                path=video_file.name,
                duration=0,  # Will be filled by processor
                frame_count=0,
                fps=0,
                resolution=(0, 0),
                filesize=os.path.getsize(video_file.name),
                has_audio=False
            )
            
            info, duration = self.processor.get_video_info(video_file.name)
            metadata.duration = duration
            GLOBAL_STATE.current_video = metadata
            
            return info
        except Exception as e:
            return f"Error reading video file: {e}"

    def _handle_processing(self, video_file: gr.File,
                         target_minutes: float) -> str:
        """Handle video processing."""
        if self.processing_lock.locked():
            return "Processing already in progress"
            
        with self.processing_lock:
            try:
                if not video_file:
                    return "No video file selected"
                    
                if not os.path.exists(video_file.name):
                    return "Selected file does not exist"
                    
                if target_minutes <= 0:
                    return "Target duration must be greater than 0"

                # Generate output path
                output_dir = "output"
                os.makedirs(output_dir, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = os.path.join(
                    output_dir,
                    f"processed_{timestamp}_{os.path.basename(video_file.name)}"
                )

                # Process video
                result = self.processor.process_video(
                    video_file.name,
                    output_path,
                    target_minutes * 60,
                    progress_callback=self._update_progress
                )

                if result:
                    return f"Processing complete!\nOutput saved to: {output_path}"
                else:
                    return "Processing failed. Check the logs for details."

            except Exception as e:
                return f"Processing error: {e}"

    def _handle_cancellation(self) -> str:
        self.processor.cancel_processing()
        self.cancel_flag.set()
        return "Processing cancelled"

    def _handle_batch_processing(self, selected_files: List[str],
                               target_duration: float) -> str:
        """Handle batch processing of multiple files with placeholder check."""
        # Check for placeholder messages
        if any(msg.startswith(("⚠️", "❌")) for msg in selected_files):
            print("Info: Invalid file selection")
            time.sleep(1)
            return "Please add files to input directory first"
        
        # Rest of the original processing logic...
        if not selected_files:
            print("Info: No files selected")
            time.sleep(1)
            return "No files selected"
        
        try:
            total_files = len(selected_files)
            for idx, file in enumerate(selected_files):
                if self.cancel_flag.is_set():
                    print("Info: Batch processing cancelled")
                    time.sleep(1)
                    return "Batch processing cancelled"
                
                input_path = os.path.join("input", file)
                output_path = os.path.join(
                    "output",
                    f"processed_{int(time.time())}_{file}"
                )
                self.batch_processor.add_to_queue(
                    input_path,
                    output_path,
                    target_duration * 60
                )
                
                # Process immediately for demonstration; in practice, use queue
                result = self.processor.process_video(
                    input_path,
                    output_path,
                    target_duration * 60,
                    progress_callback=self._update_progress
                )
                if not result:
                    print(f"Error: Failed to process {file}")
                    time.sleep(5)
                
                progress = ((idx + 1) / total_files) * 100
                self._update_progress("Batch Processing", progress, f"Processed {idx + 1}/{total_files}")
            
            print("Info: Batch processing complete")
            time.sleep(1)
            return "Batch processing complete"
        
        except Exception as e:
            print(f"Error: Batch processing error - {e}")
            time.sleep(5)  # Error message
            return f"Batch processing error: {e}"

    def _handle_batch_cancellation(self) -> str:
        """Handle batch processing cancellation."""
        self.batch_processor.cancel_processing()
        return "Batch processing cancelled"

    def _save_settings(self, motion_threshold: float, min_scene_duration: float, max_speed: float,
                      preserve_pitch: bool, enhance_audio: bool, use_opencl: bool,
                      use_avx2: bool, use_aocl: bool) -> str:
        """Save UI settings to persistent.json with strict type validation."""
        # Validate numerical parameters
        if not isinstance(motion_threshold, (int, float)) or not (0 <= motion_threshold <= 1):
            return "Motion threshold must be a number between 0 and 1"
        
        if not isinstance(min_scene_duration, (int, float)) or min_scene_duration <= 0:
            return "Minimum scene duration must be a positive number"
        
        if not isinstance(max_speed, (int, float)) or max_speed < 1:
            return "Maximum speed factor must be at least 1"

        # Validate boolean parameters
        bool_params = {
            'preserve_pitch': preserve_pitch,
            'enhance_audio': enhance_audio,
            'use_opencl': use_opencl,
            'use_avx2': use_avx2,
            'use_aocl': use_aocl
        }
        
        for param, value in bool_params.items():
            if not isinstance(value, bool):
                return f"Invalid type for {param.replace('_', ' ')} - must be boolean"

        try:
            # Create structured settings with explicit typing
            new_settings = {
                'motion_threshold': float(motion_threshold),
                'min_scene_duration': float(min_scene_duration),
                'max_speed_factor': float(max_speed),
                'preserve_pitch': bool(preserve_pitch),
                'enhance_audio': bool(enhance_audio),
                'hardware_acceleration': {
                    'use_opencl': bool(use_opencl),
                    'use_avx2': bool(use_avx2),
                    'use_aocl': bool(use_aocl)
                }
            }

            # Merge with existing config to preserve unmodified settings
            full_config = {**PROCESSING_CONFIG, **new_settings}
            full_config['hardware_acceleration'] = {
                **PROCESSING_CONFIG.get('hardware_acceleration', {}),
                **new_settings['hardware_acceleration']
            }

            # Update audio config separately
            AUDIO_CONFIG.update({
                'preserve_pitch': new_settings['preserve_pitch'],
                'enhance_audio': new_settings['enhance_audio']
            })

            # Save to persistent storage
            persistent_file = os.path.join("data", "persistent.json")
            with open(persistent_file, "w") as f:
                json.dump(full_config, f, indent=4, default=str)

            # Update runtime configurations
            PROCESSING_CONFIG.update(full_config)
            
            print("Info: Settings saved successfully")
            time.sleep(1)
            return "Settings saved successfully"
            
        except Exception as e:
            print(f"Error: Settings save failed - {e}")
            time.sleep(5)
            return f"Error saving settings: {str(e)}"

    def _reset_settings(self) -> List[Any]:
        """Reset settings to defaults."""
        return [
            self.config['motion_threshold'],  # Ensure all references use dict syntax
            self.config['min_scene_duration'],
            self.config['max_speed_factor'],
            AUDIO_CONFIG['preserve_pitch'],
            AUDIO_CONFIG['enhance_audio'],
            True,  # use_opencl
            True,  # use_avx2
            True   # use_aocl
        ]

    def _refresh_log(self) -> Tuple[str, str]:
        """Refresh log display."""
        print("Info: Logging is disabled.")
        time.sleep(1)
        metrics = self.metrics.get_metrics_report()
        return "Logging is disabled.", metrics

    def _clear_log(self) -> str:
        """Clear the log content."""
        print("Info: No logs to clear.")
        time.sleep(1)
        return "No logs to clear."

    def _get_input_files(self) -> List[str]:
        """Get valid files from input directory."""
        try:
            input_dir = os.path.abspath("input")
            if not os.path.exists(input_dir):
                os.makedirs(input_dir)
                print(f"Info: Created input directory: {input_dir}")
                time.sleep(1)
                return []  # Return empty list instead of placeholder
            
            supported_ext = tuple(ext.lower() for ext in self.file_processor.supported_formats)
            return [f for f in os.listdir(input_dir)
                    if os.path.splitext(f)[1].lower() in supported_ext]
                    
        except Exception as e:
            print(f"Error: Failed to list files in input directory - {e}")
            time.sleep(5)
            return []

    def _update_file_list(self) -> List[Dict]:
        files = self._get_input_files()
        has_files = bool(files)
        
        # Return valid component updates without placeholder messages
        return [
            gr.CheckboxGroup.update(
                choices=files if files else [],
                interactive=has_files,
                value=[]
            ),
            gr.Markdown.update(
                value="⚠️ Add videos to input directory first" if not has_files else "",
                visible=not has_files
            ),
            gr.Button.update(interactive=has_files)
        ]

    def _update_button_states(self) -> Tuple[Dict, Dict, Dict, Dict]:
        is_processing = self.processing_lock.locked()
        return (
            gr.Button.update(interactive=not is_processing),  # process_btn
            gr.Button.update(interactive=is_processing),      # cancel_btn
            gr.File.update(interactive=not is_processing),    # video_input
            gr.Number.update(interactive=not is_processing)   # target_duration
        )

    def _update_progress(self, stage: str, progress: float, message: str) -> None:
        self.progress_bar.update(progress / 100)
        self.status_output.update(f"Stage: {stage}\nProgress: {progress:.1f}%\n{message}")
        self.metrics.update_processing_metrics(stage, progress)
        self.metrics_display.update(self.metrics.get_metrics_report())

    def _validate_target_duration(self, target_duration: float) -> Tuple[bool, str]:
        """Validate target duration against video length."""
        if not self.current_video_info:
            return False, "No video selected"
            
        current_duration = self.current_video_info['duration'] / 60
        
        if target_duration >= current_duration:
            return False, (f"Target duration ({target_duration:.1f} min) must be less "
                         f"than original duration ({current_duration:.1f} min)")
                         
        if target_duration < current_duration * 0.1:
            return False, (f"Target duration ({target_duration:.1f} min) is too short. "
                         f"Minimum recommended: {(current_duration * 0.1):.1f} min")
                         
        return True, ""

    def launch(self) -> None:
        """Launch the Gradio interface."""
        interface = self.create_interface()
        interface.launch(
            server_name="127.0.0.1",  # Explicit localhost IP
            server_port=7860,
            show_error=True,
            show_api=False  # Disable automatic API page
        )

    def _create_batch_queue_display(self) -> None:
        """Create queue management display."""
        with gr.Column():
            gr.Markdown("### Processing Queue")
            self.queue_display = gr.Dataframe(
                headers=["File", "Status", "Progress"],
                datatype=["str", "str", "str"],
                value=[],  # Empty array instead of placeholder rows
                col_count=3,
                interactive=False,
                elem_id="queue_display"
            )
            self.queue_progress = gr.Progress()
            with gr.Row():
                self.queue_up_btn = gr.Button("Move Up")
                self.queue_down_btn = gr.Button("Move Down")
                self.queue_remove_btn = gr.Button("Remove")

    def _update_queue_display(self) -> None:
        """Update queue display with current status."""
        try:
            queue_data = []
            for file_info in self.batch_processor.get_queue_status():
                queue_data.append([
                    str(file_info.get('name', '')),
                    str(file_info.get('status', 'Pending')),
                    f"{file_info.get('progress', 0.0):.1f}%"
                ])
            
            # Pad with empty rows to maintain fixed row count
            while len(queue_data) < 5:
                queue_data.append(["", "", ""])
                
            self.queue_display.update(value=queue_data)
        except Exception as e:
            print(f"Queue update error: {str(e)}")
            self.error_handler.handle_error(e, "queue_display")

    def _manage_queue(self, action: str, selected_index: int) -> None:
        """Manage queue operations."""
        if not 0 <= selected_index < len(self.selected_files):
            return
            
        if action == "up" and selected_index > 0:
            self.selected_files[selected_index-1], self.selected_files[selected_index] = \
                self.selected_files[selected_index], self.selected_files[selected_index-1]
                
        elif action == "down" and selected_index < len(self.selected_files) - 1:
            self.selected_files[selected_index], self.selected_files[selected_index+1] = \
                self.selected_files[selected_index+1], self.selected_files[selected_index]
                
        elif action == "remove":
            self.selected_files.pop(selected_index)
            
        self._update_queue_display()

    def _setup_queue_handlers(self) -> None:
        """Setup event handlers for queue management."""
        self.queue_up_btn.click(
            fn=lambda x: self._manage_queue("up", x),
            inputs=[gr.Slider(minimum=0, maximum=9, step=1)],
            outputs=[self.queue_display]
        )
        
        self.queue_down_btn.click(
            fn=lambda x: self._manage_queue("down", x),
            inputs=[gr.Slider(minimum=0, maximum=9, step=1)],
            outputs=[self.queue_display]
        )
        
        self.queue_remove_btn.click(
            fn=lambda x: self._manage_queue("remove", x),
            inputs=[gr.Slider(minimum=0, maximum=9, step=1)],
            outputs=[self.queue_display]
        )

def launch_gradio_interface():
    try:
        print("\nDebug: Starting Gradio interface launch...")
        manager = InterfaceManager()
        print("Debug: InterfaceManager created, launching interface...")
        manager.launch()
        print("Debug: Interface launched successfully")
        while True:  # Keep the main thread alive
            time.sleep(1)
    except Exception as e:
        print(f"Error: Failed to launch interface - {str(e)}")
        print(f"Debug: Full traceback:\n{traceback.format_exc()}")
        time.sleep(5)
        sys.exit(1)

if __name__ == "__main__":
    try:
        print("\nLaunching Movie Consolidator interface...")
        time.sleep(1)
        launch_gradio_interface()  # Remove log_manager
    except KeyboardInterrupt:
        print("\nShutdown requested by user")
        time.sleep(1)
        sys.exit(0)
    except Exception as e:
        print(f"\nError: Unexpected error - {e}")
        time.sleep(5)
        sys.exit(1)
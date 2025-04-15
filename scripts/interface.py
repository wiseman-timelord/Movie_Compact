# Script: `.\scripts\interface.py`

# Imports
import os, json, time, sys, traceback, webbrowser
from threading import Timer
import gradio as gr
import pandas as pd  # Add for DataFrame creation
import psutil  # Add for memory monitoring
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
    AUDIO_CONFIG
)
from scripts.analyze import VideoAnalyzer
from scripts.process import VideoProcessor, BatchProcessor

# Classes...
class InterfaceManager:
    def __init__(self):
        self.core = CoreUtilities()
        self.file_processor = FileProcessor(
            supported_formats=self.config['supported_formats']
        )
        self.settings = settings
        self.config = config
        self.gpu_selection = None
        self.use_avx2 = None
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

    def _update_file_sizes_plot(self) -> pd.DataFrame:
        """Generate DataFrame for all files in input directory."""
        files = self._get_input_files()  # Get all valid files
        if not files:
            return pd.DataFrame({"File": [], "Size (GB)": []})
        files = sorted(files)  # Alphanumeric order
        sizes = []
        for file in files:
            path = os.path.join("input", file)
            if os.path.exists(path):
                size_gb = os.path.getsize(path) / (1024**3)
                sizes.append(size_gb)
            else:
                sizes.append(0)
        return pd.DataFrame({"File": files, "Size (GB)": sizes})

    def _create_main_tab(self) -> None:
        with gr.Tab("Process Video"):
            with gr.Row():
                with gr.Column(scale=2):
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

                    # Add memory usage bar plot
                    with gr.Group():
                        gr.Markdown("### Memory Usage During Processing")
                        self.memory_bar_plot = gr.BarPlot(
                            x="Memory Type",
                            y="Usage (GB)",
                            title="Memory Usage",
                            tooltip=["Memory Type", "Usage (GB)"],
                            height=300,
                            width=400
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
                    
                    # file sizes bar plot
                    with gr.Group():
                        gr.Markdown("### File Sizes in Input Directory")  # Updated title
                        initial_file_sizes = self._update_file_sizes_plot()
                        self.file_size_bar_plot = gr.BarPlot(
                            x="File",
                            y="Size (GB)",
                            title="File Sizes",
                            tooltip=["File", "Size (GB)"],
                            height=300,
                            width=400,
                            value=initial_file_sizes  # Set initial data
                        )
                    
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
                    self._create_batch_queue_display()
            

    def _create_settings_tab(self) -> None:
        with gr.Tab("Configuration"):
            with gr.Row():
                # Video Settings Column
                with gr.Column():
                    gr.Markdown("### Video Processing")
                    self.motion_threshold = gr.Slider(
                        minimum=0.1,
                        maximum=1.0,
                        value=ConfigManager.get('processing', 'scene_detection.motion_threshold', 0.3),
                        label="Motion Detection Sensitivity"
                    )
                    self.min_scene_duration = gr.Number(
                        value=ConfigManager.get('processing', 'scene_detection.min_scene_duration', 2.0),
                        label="Minimum Scene Duration (seconds)",
                        precision=1
                    )
                    self.max_speed = gr.Slider(
                        minimum=1.0,
                        maximum=8.0,
                        value=ConfigManager.get('speed', 'max_speed_factor', 4.0),
                        label="Maximum Speed Factor"
                    )

                # Hardware Configuration Column
                with gr.Column():
                    gr.Markdown("### Hardware Configuration")
                    
                    # Manual VRAM Selection
                    self.vram_dropdown = gr.Dropdown(
                        label="VRAM Allocation (GB)",
                        choices=ConfigManager.get('hardware', 'vram_options', ['8']),
                        value=ConfigManager.get('hardware', 'selected_vram', '8'),
                        interactive=True
                    )
                    
                    # Acceleration Status
                    gr.Markdown("### Active Acceleration", elem_id="accel_status")
                    self.opencl_status = gr.Markdown(
                        f"OpenCL: {'Enabled' if self.hardware_capabilities['OpenCL'] else 'Disabled'}"
                    )
                    self.avx2_status = gr.Markdown(
                        f"AVX2: {'Available' if self.hardware_capabilities['AVX2'] else 'Unavailable'}"
                    )

            # Audio Settings Row
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Audio Processing")
                    self.preserve_pitch = gr.Checkbox(
                        value=ConfigManager.get('audio', 'preserve_pitch', True),
                        label="Preserve Audio Pitch During Speed Changes"
                    )
                    self.enhance_audio = gr.Checkbox(
                        value=ConfigManager.get('audio', 'enhance_audio', True),
                        label="Enable Audio Enhancement"
                    )

            # Control Buttons
            with gr.Row():
                self.save_settings_btn = gr.Button("Save Settings", variant="primary")
                self.reset_settings_btn = gr.Button("Reset to Defaults", variant="secondary")

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

    def _update_progress(self, stage: str, progress: float, message: str) -> None:
        """Update progress bar, status, metrics, and memory usage."""
        with self.processing_lock:
            # Update progress and status
            self.progress_bar.update(progress / 100)
            self.status_output.update(f"Stage: {stage}\nProgress: {progress:.1f}%\n{message}")
            
            # Update metrics
            self.metrics.update_processing_metrics(stage, progress)
            self.metrics_display.update(self.metrics.get_metrics_report())
            
            # Update memory plot every 5 seconds
            current_time = time.time()
            if not hasattr(self, 'last_memory_update') or current_time - self.last_memory_update > 5:
                self.last_memory_update = current_time
                memory_df = self._get_memory_usage_df()
                self.memory_bar_plot.update(value=memory_df)

    def _get_memory_usage_df(self) -> pd.DataFrame:
        """Generate DataFrame for memory usage bar plot."""
        process = psutil.Process()
        mem_info = process.memory_info()
        ram_gb = mem_info.rss / (1024**3)  # Resident Set Size (RAM)
        pagefile_gb = getattr(mem_info, 'pagefile', 0) / (1024**3) if hasattr(mem_info, 'pagefile') else 0  # Page File on Windows
        return pd.DataFrame({
            "Memory Type": ["RAM", "Page File"],
            "Usage (GB)": [ram_gb, pagefile_gb]
        })

    def _setup_event_handlers(self, interface: gr.Blocks) -> None:
        """Setup event handlers for interface elements."""
        # Existing handlers remain unchanged, only listing for completeness
        self.refresh_files_btn.click(
            fn=self._update_file_list,
            inputs=None,
            outputs=[self.file_list, self.batch_status_msg, self.batch_process_btn]
        ).then(
            fn=self._update_file_sizes_plot,  # Add this line
            inputs=None,
            outputs=self.file_size_bar_plot
        )

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

        self.batch_process_btn.click(
            fn=self._handle_batch_processing,
            inputs=[self.file_list, self.batch_target_duration],
            outputs=[self.batch_status],
            preprocess=False
        )

        self.batch_cancel_btn.click(
            fn=self._handle_batch_cancellation,
            inputs=None,
            outputs=[self.batch_status],
            queue=False
        )

        self.save_settings_btn.click(
            fn=self._save_settings,
            inputs=[
                self.vram_dropdown,
                self.motion_threshold,
                self.min_scene_duration,
                self.max_speed,
                self.preserve_pitch,
                self.enhance_audio
            ],
            outputs=[self.status_output]
        )
        
        self.reset_settings_btn.click(
            fn=self._reset_settings,
            inputs=None,
            outputs=[
                self.vram_dropdown,
                self.motion_threshold,
                self.min_scene_duration,
                self.max_speed,
                self.preserve_pitch,
                self.enhance_audio
            ]
        )

        self.vram_dropdown.change(
            fn=lambda v: gr.Markdown.update(
                value=f"VRAM Allocation: {v}GB | Restart required for full effect"
            ),
            inputs=self.vram_dropdown,
            outputs=self.opencl_status
        )

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

    def _save_settings(self, vram_gb: str, motion_thresh: float, min_duration: float,
                      max_speed: float, preserve_pitch: bool, enhance_audio: bool):
        """Handle settings save with validation"""
        try:
            # Validate numerical parameters
            if not (0.1 <= motion_thresh <= 1.0):
                raise ValueError("Motion threshold must be between 0.1 and 1.0")
                
            if min_duration < 0.5:
                raise ValueError("Minimum scene duration must be at least 0.5 seconds")
                
            if max_speed < 1.0 or max_speed > 8.0:
                raise ValueError("Speed factor must be between 1.0 and 8.0")
                
            # Validate VRAM selection
            valid_vram = ConfigManager.get('hardware', 'vram_options', ['8'])
            if vram_gb not in valid_vram:
                raise ValueError(f"Invalid VRAM selection. Valid options: {', '.join(valid_vram)}")

            # Update configuration through ConfigManager
            ConfigManager.update('processing', {
                'scene_detection': {
                    'motion_threshold': float(motion_thresh),
                    'min_scene_duration': float(min_duration)
                },
                'speed': {
                    'max_speed_factor': float(max_speed)
                }
            })
            
            ConfigManager.update('audio', {
                'preserve_pitch': bool(preserve_pitch),
                'enhance_audio': bool(enhance_audio)
            })
            
            ConfigManager.set_vram(vram_gb)

            # Persist to disk
            persistent_path = os.path.join(BASE_DIR, 'data', 'persistent.json')
            with open(persistent_path, 'w') as f:
                json.dump({
                    'processing': ConfigManager._configs['processing'],
                    'audio': ConfigManager._configs['audio'],
                    'hardware': ConfigManager._configs['hardware']
                }, f, indent=4)

            # Update runtime components
            self.processor.config = ConfigManager._configs['processing']
            self.processor.audio_config = ConfigManager._configs['audio']
            
            return "Settings saved successfully"
            
        except Exception as e:
            self.error_handler.handle_error(e, "settings_save")
            return f"Error saving settings: {str(e)}"

    def _reset_settings(self):
        """Reset all settings to defaults"""
        try:
            # Hardware defaults
            ConfigManager.set_vram('8')
            ConfigManager.update('hardware', {
                'opencl_enabled': True,
                'avx2_fallback': True
            })
            
            # Processing defaults
            ConfigManager.update('processing', {
                'scene_detection': {
                    'motion_threshold': 0.3,
                    'min_scene_duration': 2.0
                },
                'speed': {
                    'max_speed_factor': 4.0
                }
            })
            
            # Audio defaults
            ConfigManager.update('audio', {
                'preserve_pitch': True,
                'enhance_audio': True
            })
            
            # Persist defaults
            persistent_path = os.path.join(BASE_DIR, 'data', 'persistent.json')
            if os.path.exists(persistent_path):
                os.remove(persistent_path)
            ConfigManager.load_persistent()
            
            return [
                '8',  # vram_dropdown
                0.3,  # motion_threshold
                2.0,  # min_scene_duration
                4.0,  # max_speed
                True,  # preserve_pitch
                True   # enhance_audio
            ]
            
        except Exception as e:
            self.error_handler.handle_error(e, "settings_reset")
            return [
                gr.Dropdown.update(value='8'),
                gr.Slider.update(value=0.3),
                gr.Number.update(value=2.0),
                gr.Slider.update(value=4.0),
                gr.Checkbox.update(value=True),
                gr.Checkbox.update(value=True)
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
        while True:
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
        launch_gradio_interface()
    except KeyboardInterrupt:
        print("\nShutdown requested by user")
        time.sleep(1)
        sys.exit(0)
    except Exception as e:
        print(f"\nError: Unexpected error - {e}")
        time.sleep(5)
        sys.exit(1)
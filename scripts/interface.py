# .\scripts\interface.py

import os
import json
import gradio as gr
import sys
from typing import Dict, Any, Optional, Tuple
from utility import (
    load_hardware_config,
    load_settings,
    log_event,
    log_manager,
    MetricsCollector
)
from process import VideoProcessor

class MovieConsolidatorError(Exception):
    """Base exception for Movie Consolidator errors."""
    pass

class ProcessingError(MovieConsolidatorError):
    """Raised when video processing fails."""
    pass

class AnalysisError(MovieConsolidatorError):
    """Raised when video analysis fails."""
    pass

class HardwareError(MovieConsolidatorError):
    """Raised when hardware-related operations fail."""
    pass

class ConfigurationError(MovieConsolidatorError):
    """Raised when configuration-related operations fail."""
    pass

class GradioInterface:
    def __init__(self):
        self.settings = load_settings()
        self.hardware_config = load_hardware_config()
        self.video_config = self.settings.get('video', {})
        self.metrics = MetricsCollector()
        self.processor = VideoProcessor()
        self.active_process = False

    def load_persistent_settings(self) -> Dict[str, Any]:
        """Load settings from persistent.json."""
        try:
            persistent_file = os.path.join("data", "persistent.json")
            if os.path.exists(persistent_file):
                with open(persistent_file, "r") as f:
                    return json.load(f)
            return {}
        except Exception as e:
            log_event(f"Error loading persistent settings: {e}", "ERROR", "CONFIG")
            return {}

    def save_persistent_settings(self, settings: Dict[str, Any]) -> None:
        """Save settings to persistent.json."""
        try:
            persistent_file = os.path.join("data", "persistent.json")
            with open(persistent_file, "w") as f:
                json.dump(settings, f, indent=4)
            log_event("Settings saved successfully", "INFO", "CONFIG")
        except Exception as e:
            log_event(f"Error saving settings: {e}", "ERROR", "CONFIG")

    def update_event_log(self) -> str:
        """Get recent logs for display."""
        logs = log_manager.get_recent_logs(num_lines=20)
        return "\n".join(logs)

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
            clip.close()

            info = (
                f"Video Information:\n"
                f"Duration: {duration:.1f} seconds ({duration/60:.1f} minutes)\n"
                f"Size: {size_mb:.1f} MB\n"
                f"FPS: {fps}\n"
                f"Resolution: {resolution}"
            )
            return info, duration
        except Exception as e:
            log_event(f"Error getting video info: {e}", "ERROR", "INFO")
            return "Error reading video file", 0.0

    def process_video(self, video_path: str, target_minutes: float) -> str:
        """Process a single video file."""
        try:
            self.active_process = True
            if not video_path:
                return "No video file selected."

            output_dir = self.settings.get("paths", {}).get("output_path", "output")
            output_path = os.path.join(
                output_dir, 
                f"processed_{os.path.basename(video_path)}"
            )

            log_event("Starting video processing", "INFO", "PROCESSING")
            self.metrics.start_phase_timing("processing")

            # Convert minutes to seconds for processing
            target_duration = target_minutes * 60
            
            result = self.processor.process_video(video_path, output_path, target_duration)
            
            self.metrics.end_phase_timing("processing")
            self.active_process = False
            
            if result:
                return (f"Processing complete. Output saved to: {output_path}\n\n"
                       f"{self.metrics.get_metrics_report()}")
            else:
                return "Processing failed. Check the logs for details."
            
        except Exception as e:
            error_msg = f"Processing failed: {e}"
            log_event(error_msg, "ERROR", "PROCESSING")
            self.active_process = False
            return error_msg

    def launch_interface(self):
        """Launch the Gradio interface."""
        settings = self.load_persistent_settings()
        
        with gr.Blocks(title="Movie Consolidator") as interface:
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
                                    file_types=["video"]
                                )
                                video_info = gr.TextArea(
                                    label="Video Information",
                                    interactive=False,
                                    lines=5
                                )
                                target_duration = gr.Number(
                                    label="Target Duration (minutes)",
                                    value=30,
                                    minimum=1,
                                    info="Desired length of the final video in minutes"
                                )
                            
                            # Process Section
                            with gr.Group():
                                gr.Markdown("### Processing Controls")
                                process_button = gr.Button(
                                    "Process Video", 
                                    variant="primary",
                                    scale=2
                                )
                                current_phase = gr.Textbox(
                                    label="Current Processing Phase",
                                    interactive=False
                                )
                                progress_bar = gr.Progress()
                                status_output = gr.TextArea(
                                    label="Processing Status",
                                    interactive=False,
                                    lines=10
                                )

                        # Monitoring Column
                        with gr.Column(scale=1):
                            # Event Log
                            gr.Markdown("### Live Processing Log")
                            event_log = gr.TextArea(
                                label="Recent Events",
                                value=self.update_event_log(),
                                interactive=False,
                                lines=15,
                                autoscroll=True
                            )
                            
                            # Metrics Display
                            gr.Markdown("### Processing Metrics")
                            metrics_display = gr.TextArea(
                                label="Current Metrics",
                                interactive=False,
                                lines=10
                            )
                            
                            refresh_log = gr.Button("Refresh Displays")
                
                # Configuration Tab
                with gr.Tab("Settings"):
                    gr.Markdown("## Processing Settings")
                    
                    with gr.Row():
                        # Hardware Info
                        with gr.Column():
                            gr.Markdown("### Hardware Configuration")
                            hardware_info = gr.JSON(
                                label="Available Hardware Features",
                                value=self.hardware_config,
                                interactive=False
                            )
                        
                        # Video Settings
                        with gr.Column():
                            gr.Markdown("### Video Settings")
                            preview_height = gr.Slider(
                                label="Preview Resolution Height",
                                minimum=240,
                                maximum=720,
                                step=120,
                                value=settings.get("video_settings", {}).get("preview_height", 360),
                                info="Height of preview video used for analysis"
                            )

                    with gr.Row():
                        # Detection Settings
                        with gr.Column():
                            gr.Markdown("### Detection Settings")
                            motion_threshold = gr.Slider(
                                label="Motion Detection Sensitivity",
                                minimum=0,
                                maximum=1,
                                step=0.1,
                                value=settings.get("motion_threshold", 0.5),
                                info="Lower values detect more subtle movements"
                            )
                            static_threshold = gr.Slider(
                                label="Static Frame Threshold",
                                minimum=0.9,
                                maximum=1.0,
                                step=0.01,
                                value=settings.get("static_threshold", 0.99),
                                info="Higher values more aggressively detect static frames"
                            )

                        # Speed Settings
                        with gr.Column():
                            gr.Markdown("### Speed Settings")
                            max_speed = gr.Slider(
                                label="Maximum Speed Factor",
                                minimum=1.0,
                                maximum=8.0,
                                step=0.5,
                                value=settings.get("speed_settings", {}).get("max_speed_factor", 4.0),
                                info="Maximum speed multiplier for non-action scenes"
                            )

                    # Save/Reset Buttons
                    with gr.Row():
                        save_button = gr.Button("Save Settings", variant="primary")
                        reset_button = gr.Button("Reset to Default")

                # Event handlers
                video_input.change(
                    fn=lambda x: self.get_video_info(x.name) if x else ("No file selected", 0),
                    inputs=[video_input],
                    outputs=[video_info]
                )
                
                process_button.click(
                    fn=self.process_video,
                    inputs=[video_input, target_duration],
                    outputs=[status_output]
                )

                refresh_log.click(
                    fn=lambda: (
                        self.update_event_log(),
                        self.update_metrics_display()
                    ),
                    outputs=[event_log, metrics_display]
                )
                
                # Settings handlers
                save_button.click(
                    fn=lambda *args: self.save_persistent_settings({
                        "video_settings": {
                            "preview_height": args[0]
                        },
                        "motion_threshold": args[1],
                        "static_threshold": args[2],
                        "speed_settings": {
                            "max_speed_factor": args[3]
                        }
                    }),
                    inputs=[
                        preview_height,
                        motion_threshold,
                        static_threshold,
                        max_speed
                    ]
                )
                
                # Reset to defaults
                reset_button.click(
                    fn=lambda: {
                        "video_settings": {
                            "preview_height": 360
                        },
                        "motion_threshold": 0.5,
                        "static_threshold": 0.99,
                        "speed_settings": {
                            "max_speed_factor": 4.0
                        }
                    },
                    outputs=[
                        preview_height,
                        motion_threshold,
                        static_threshold,
                        max_speed
                    ]
                )
            
            # Auto-refresh displays every 2 seconds
            gr.on(
                lambda: gr.Timer(2), 
                lambda: (
                    self.update_event_log(),
                    self.update_metrics_display()
                ),
                outputs=[event_log, metrics_display]
            )
            
            # Launch in browser
            interface.launch(inbrowser=True)

def launch_gradio_interface():
    """Launch the Gradio interface."""
    interface = GradioInterface()
    interface.launch_interface()

if __name__ == "__main__":
    launch_gradio_interface()
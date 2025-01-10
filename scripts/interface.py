# .\scripts\interface.py

# .\scripts\interface.py

import os, json, gradio as gr, sys
from typing import Dict, List, Tuple, Any, Optional
from utility import (
    load_hardware_config,
    load_settings,
    get_video_files,
    get_video_duration,
    log_event
)
from scripts.generate import process_videos, process_video

class GradioInterface:
    def __init__(self):
        self.settings = load_settings()
        self.hardware_config = load_hardware_config()
        self.search_criteria = self.settings.get('search', {})
        self.video_config = self.settings.get('video', {})

    def load_persistent_settings(self) -> Dict[str, Any]:
        """Load settings from persistent.json."""
        try:
            persistent_file = os.path.join("data", "persistent.json")
            if os.path.exists(persistent_file):
                with open(persistent_file, "r") as f:
                    return json.load(f)
            return {}
        except Exception as e:
            log_event(f"Error loading persistent settings: {e}")
            return {}

    def save_persistent_settings(self, settings: Dict[str, Any]) -> None:
        """Save settings to persistent.json."""
        try:
            persistent_file = os.path.join("data", "persistent.json")
            with open(persistent_file, "w") as f:
                json.dump(settings, f, indent=4)
            log_event("Settings saved successfully")
        except Exception as e:
            log_event(f"Error saving settings: {e}")

    def update_event_log(self) -> str:
        """Read and return the event log content."""
        try:
            log_file = os.path.join("data", "events.txt")
            if os.path.exists(log_file):
                with open(log_file, "r") as f:
                    lines = f.readlines()
                    # Return last 20 lines for display
                    return "".join(lines[-20:])
            return "No events logged yet."
        except Exception as e:
            return f"Error reading event log: {e}"

    def parse_folder_paths(self, folder_paths_str: str) -> List[str]:
        """Parse semicolon-separated folder paths."""
        return [path.strip() for path in folder_paths_str.split(';') if path.strip()]

    def list_video_files(self, folders: List[str]) -> List[Dict[str, Any]]:
        """List video files with metadata."""
        video_files = []
        supported_formats = self.video_config.get('supported_formats', ['.mp4', '.avi', '.mkv'])
        
        try:
            for folder in folders:
                if not os.path.exists(folder):
                    log_event(f"Folder not found: {folder}")
                    continue
                    
                for root, _, files in os.walk(folder):
                    for file in files:
                        if os.path.splitext(file)[1].lower() in supported_formats:
                            full_path = os.path.join(root, file)
                            duration = get_video_duration(full_path)
                            video_files.append({
                                'path': full_path,
                                'name': file,
                                'duration': duration,
                                'size': os.path.getsize(full_path)
                            })
            
            log_event(f"Found {len(video_files)} video files")
            return video_files
            
        except Exception as e:
            log_event(f"Error listing video files: {e}")
            return []

    def format_video_list(self, videos: List[Dict[str, Any]]) -> Tuple[str, float, int]:
        """Format video list for display."""
        display_list = []
        total_duration = 0
        
        for video in videos:
            name = video['name']
            if len(name) > 40:
                name = name[:37] + "..."
            duration = video['duration']
            size_mb = video['size'] / (1024 * 1024)
            
            display_list.append(f"{name} - {duration:.1f}s - {size_mb:.1f}MB")
            total_duration += duration
            
        return "\n".join(display_list), total_duration, len(videos)

    def process_videos_interface(self, folder_paths_str: str) -> str:
        """Process videos from interface."""
        try:
            folders = self.parse_folder_paths(folder_paths_str)
            if not folders:
                return "No folders specified."
                
            log_event("Starting video processing from interface")
            settings = self.load_persistent_settings()
            output_dir = settings.get("paths", {}).get("output_path", "output")
            
            # Process each folder
            for folder in folders:
                if not os.path.exists(folder):
                    log_event(f"Folder not found: {folder}")
                    continue
                    
                process_videos(folder, output_dir)
            
            return f"Processing complete. Results saved to {output_dir}"
            
        except Exception as e:
            error_msg = f"Processing failed: {e}"
            log_event(error_msg)
            return error_msg

    def launch_interface(self):
        """Launch the Gradio interface."""
        settings = self.load_persistent_settings()
        
        with gr.Blocks(title="Movie Consolidator") as interface:
            with gr.Tabs():
                # Consolidation Tab
                with gr.Tab("Consolidation"):
                    with gr.Row():
                        with gr.Column(scale=2):
                            gr.Markdown("# Video Consolidation")
                            
                            # Input Section
                            with gr.Group():
                                folder_input = gr.Textbox(
                                    label="Input Folders (semicolon-separated)",
                                    placeholder="Enter folder paths here"
                                )
                                list_button = gr.Button("List Videos")
                                video_list = gr.Textbox(label="Selected Videos", interactive=False)
                                total_duration = gr.Number(label="Total Duration (seconds)", interactive=False)
                                file_count = gr.Number(label="Number of Files", interactive=False)
                            
                            # Process Section
                            with gr.Group():
                                process_button = gr.Button("Process Videos", variant="primary")
                                current_phase = gr.Textbox(label="Current Phase", interactive=False)
                                progress_bar = gr.Progress()
                                status_output = gr.Textbox(label="Status", interactive=False)

                        # Event Log Column
                        with gr.Column(scale=1):
                            event_log = gr.Textbox(
                                label="Processing Log",
                                value=self.update_event_log(),
                                interactive=False,
                                max_lines=20,
                                autoscroll=True
                            )
                            refresh_log = gr.Button("Refresh Log")
                
                # Configuration Tab
                with gr.Tab("Configuration"):
                    gr.Markdown("# Settings")
                    
                    with gr.Row():
                        # Hardware Info
                        with gr.Column():
                            hardware_info = gr.JSON(
                                label="Hardware Features",
                                value=self.hardware_config
                            )
                        
                        # Video Settings
                        with gr.Column():
                            target_length = gr.Number(
                                label="Target Length (minutes)",
                                value=settings.get("video_settings", {}).get("target_length", 30)
                            )
                            preview_height = gr.Slider(
                                label="Preview Height",
                                minimum=240,
                                maximum=720,
                                step=120,
                                value=settings.get("video_settings", {}).get("preview_height", 360)
                            )

                    with gr.Row():
                        # Detection Settings
                        with gr.Column():
                            gr.Markdown("### Detection Settings")
                            motion_threshold = gr.Slider(
                                label="Motion Threshold",
                                minimum=0,
                                maximum=1,
                                step=0.1,
                                value=settings.get("motion_threshold", 0.5)
                            )
                            texture_threshold = gr.Slider(
                                label="Texture Threshold",
                                minimum=0,
                                maximum=1,
                                step=0.1,
                                value=settings.get("texture_threshold", 0.6)
                            )
                            static_threshold = gr.Slider(
                                label="Static Frame Threshold",
                                minimum=0.9,
                                maximum=1.0,
                                step=0.01,
                                value=settings.get("static_threshold", 0.99)
                            )

                        # Scene Settings
                        with gr.Column():
                            gr.Markdown("### Scene Settings")
                            min_scene_length = gr.Number(
                                label="Minimum Scene Length (seconds)",
                                value=settings.get("scene_settings", {}).get("min_scene_length", 2)
                            )
                            max_scene_length = gr.Number(
                                label="Maximum Scene Length (seconds)",
                                value=settings.get("scene_settings", {}).get("max_scene_length", 300)
                            )
                            scene_threshold = gr.Slider(
                                label="Scene Change Threshold",
                                minimum=10,
                                maximum=100,
                                step=5,
                                value=settings.get("scene_settings", {}).get("scene_threshold", 30)
                            )

                    with gr.Row():
                        # Processing Settings
                        with gr.Column():
                            gr.Markdown("### Processing Settings")
                            use_gpu = gr.Checkbox(
                                label="Use GPU Acceleration",
                                value=settings.get("processor_settings", {}).get("use_gpu", True)
                            )
                            cpu_cores = gr.Slider(
                                label="CPU Threads",
                                minimum=1,
                                maximum=16,
                                step=1,
                                value=settings.get("processor_settings", {}).get("cpu_cores", 4)
                            )
                            batch_size = gr.Slider(
                                label="GPU Batch Size",
                                minimum=16,
                                maximum=128,
                                step=16,
                                value=settings.get("processor_settings", {}).get("gpu_batch_size", 32)
                            )
                        
                        # Speed Settings
                        with gr.Column():
                            gr.Markdown("### Speed Settings")
                            max_speed = gr.Slider(
                                label="Maximum Speed Factor",
                                minimum=1.0,
                                maximum=8.0,
                                step=0.5,
                                value=settings.get("speed_settings", {}).get("max_speed_factor", 4.0)
                            )
                            transition_frames = gr.Slider(
                                label="Speed Transition Frames",
                                minimum=10,
                                maximum=60,
                                step=5,
                                value=settings.get("speed_settings", {}).get("transition_frames", 30)
                            )

                    # Save/Reset Buttons
                    with gr.Row():
                        save_button = gr.Button("Save Settings", variant="primary")
                        reset_button = gr.Button("Reset to Default")
                
                # Event handlers
                list_button.click(
                    fn=lambda x: self.format_video_list(self.list_video_files(self.parse_folder_paths(x))),
                    inputs=[folder_input],
                    outputs=[video_list, total_duration, file_count]
                )
                
                process_button.click(
                    fn=self.process_videos_interface,
                    inputs=[folder_input],
                    outputs=[status_output]
                )

                refresh_log.click(
                    fn=self.update_event_log,
                    outputs=[event_log]
                )
                
                # Save all settings
                save_button.click(
                    fn=lambda *args: self.save_persistent_settings({
                        "video_settings": {
                            "target_length": args[0],
                            "preview_height": args[1]
                        },
                        "motion_threshold": args[2],
                        "texture_threshold": args[3],
                        "static_threshold": args[4],
                        "scene_settings": {
                            "min_scene_length": args[5],
                            "max_scene_length": args[6],
                            "scene_threshold": args[7]
                        },
                        "processor_settings": {
                            "use_gpu": args[8],
                            "cpu_cores": args[9],
                            "gpu_batch_size": args[10]
                        },
                        "speed_settings": {
                            "max_speed_factor": args[11],
                            "transition_frames": args[12]
                        }
                    }),
                    inputs=[
                        target_length, preview_height,
                        motion_threshold, texture_threshold, static_threshold,
                        min_scene_length, max_scene_length, scene_threshold,
                        use_gpu, cpu_cores, batch_size,
                        max_speed, transition_frames
                    ]
                )
                
                # Reset to defaults
                reset_button.click(
                    fn=lambda: {
                        "video_settings": {
                            "target_length": 30,
                            "preview_height": 360
                        },
                        "motion_threshold": 0.5,
                        "texture_threshold": 0.6,
                        "static_threshold": 0.99,
                        "scene_settings": {
                            "min_scene_length": 2,
                            "max_scene_length": 300,
                            "scene_threshold": 30
                        },
                        "processor_settings": {
                            "use_gpu": True,
                            "cpu_cores": 4,
                            "gpu_batch_size": 32
                        },
                        "speed_settings": {
                            "max_speed_factor": 4.0,
                            "transition_frames": 30
                        }
                    },
                    outputs=[
                        target_length, preview_height,
                        motion_threshold, texture_threshold, static_threshold,
                        min_scene_length, max_scene_length, scene_threshold,
                        use_gpu, cpu_cores, batch_size,
                        max_speed, transition_frames
                    ]
                )
            
            # Auto-refresh event log every 5 seconds
            gr.on(lambda: gr.Timer(5), self.update_event_log, outputs=[event_log])
            
            # Launch in browser
            interface.launch(inbrowser=True)

def launch_gradio_interface():
    """Launch the Gradio interface."""
    interface = GradioInterface()
    interface.launch_interface()

if __name__ == "__main__":
    launch_gradio_interface()
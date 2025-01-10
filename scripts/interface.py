# .\scripts\interface.py

import os
import json
import gradio as gr
import sys
from utility import get_video_duration, load_hardware_config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def load_hardware_config():
    hardware_file = os.path.join("data", "hardware.txt")
    hardware_config = {
        "x64": False,
        "Avx2": False,
        "Aocl": False,
        "OpenCL": False,
    }
    if os.path.exists(hardware_file):
        with open(hardware_file, "r") as f:
            for line in f:
                key, value = line.strip().split(": ")
                hardware_config[key] = value.lower() == "true"
    return hardware_config


def load_persistent_settings():
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    persistent_file = os.path.join(data_dir, "persistent.json")
    if not os.path.exists(persistent_file):
        return {}
    with open(persistent_file, "r") as f:
        return json.load(f)

def save_persistent_settings(settings):
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    with open(os.path.join(data_dir, "persistent.json"), "w") as f:
        json.dump(settings, f, indent=4)

def parse_folder_paths(folder_paths_str):
    """
    Parse semicolon-separated folder paths into a list of paths.
    """
    return [path.strip() for path in folder_paths_str.split(';') if path.strip()]

def list_video_files(folders):
    """
    List all video files in the specified folders.
    Supported formats: .mp4, .avi, .mkv.
    """
    video_extensions = ['.mp4', '.avi', '.mkv']
    video_files = []
    for folder in folders:
        for root, _, files in os.walk(folder):
            for file in files:
                if os.path.splitext(file)[1].lower() in video_extensions:
                    video_files.append(os.path.join(root, file))
    return video_files

def display_video_info(video_files):
    video_extensions = ['.mp4', '.avi', '.mkv']
    display_list = []
    total_duration = 0
    for path in video_files:
        filename = os.path.basename(path)
        if len(filename) > 14:
            filename = filename[:11] + '...'
        duration = get_video_duration(path)
        display_list.append(f"{filename} - {duration:.2f} seconds")
        total_duration += duration
    return display_list, total_duration, len(video_files)

def get_video_duration(file_path):
    """
    Get the duration of a video file using moviepy.
    """
    import moviepy.editor as mp
    clip = mp.VideoFileClip(file_path)
    duration = clip.duration
    clip.close()
    return duration

def update_keys(
    new_motion_threshold=None,
    new_texture_threshold=None,
    new_audio_threshold=None,
    new_keywords=None,
    new_image_size=None,
    new_target_length=None,
    new_output_quality=None,
    new_use_gpu=None,  # Add OpenCL toggle
    new_cpu_cores=None
):
    settings = load_persistent_settings()
    if new_use_gpu is not None:
        settings["processor_settings"]["use_gpu"] = new_use_gpu
    save_persistent_settings(settings)
    return "Settings updated successfully!"

def reset_settings():
    """
    Reset persistent settings to default values.
    """
    default_settings = {
        "motion_threshold": 0.5,
        "texture_threshold": 0.6,
        "audio_threshold": 0.7,
        "text_keywords": [],
        "image_size": "900p",
        "target_length": 5,
        "output_quality": "720p",
        "processor_settings": {
            "use_gpu": True,
            "cpu_cores": 4
        }
    }
    save_persistent_settings(default_settings)
    return "Settings reset to default."

def process_videos_interface(folder_paths_str, target_length, output_quality):
    """
    Process videos based on user input and generate a summary.
    """
    from scripts.generate import process_videos
    folders = parse_folder_paths(folder_paths_str)
    video_files = list_video_files(folders)
    if not video_files:
        return "No video files selected."
    output_path = 'output_summary.mp4'
    process_videos(video_files, output_path, target_length, {"resolution": output_quality})
    return output_path

def launch_gradio_interface():
    settings = load_persistent_settings()
    input_path = settings.get("input_path", "input")
    output_path = settings.get("output_path", "output")
    with gr.Blocks() as interface:
        with gr.Tabs():
            with gr.Tab("Consolidation"):
                gr.Markdown("# Video Summarization Interface")
                folder_input = gr.Textbox(label="Folder Paths (semicolon-separated)", placeholder="Enter folder paths here")
                list_button = gr.Button("List Videos")
                video_list = gr.Textbox(label="Selected Videos", interactive=False)
                total_time = gr.Textbox(label="Total Unprocessed Time", interactive=False)
                source_videos = gr.Textbox(label="Number of Source Videos", interactive=False)
                process_button = gr.Button("Process Videos")
                output_video = gr.Video(label="Output Video")
                exit_button = gr.Button("Exit Program")
                
                list_button.click(
                    fn=lambda x: "\n".join(display_video_info([x])[0]),
                    inputs=folder_input,
                    outputs=video_list
                )
                
                process_button.click(
                    fn=lambda: "Processing started...",
                    inputs=None,
                    outputs=output_video
                )
                
                exit_button.click(
                    fn=lambda: os._exit(0),
                    inputs=None,
                    outputs=None
                )
                
            with gr.Tab("Configuration"):
                gr.Markdown("# Configuration Settings")
                input_folder = gr.Folder(label="Input Folder", value=input_path)
                output_folder = gr.Folder(label="Output Folder", value=output_path)
                save_button = gr.Button("Save Configuration")
                with gr.Row():
                    with gr.Column():
                    hardware_info = gr.Textbox(
                        label="Hardware Features",
                        value=load_hardware_config(),
                        interactive=False,
                    )
                    with gr.Column():
                        motion_threshold_slider = gr.Slider(label="Motion Threshold", minimum=0, maximum=1, step=0.1, value=settings.get("motion_threshold", 0.5))
                        texture_threshold_slider = gr.Slider(label="Texture Threshold", minimum=0, maximum=1, step=0.1, value=settings.get("texture_threshold", 0.6))
                        audio_threshold_slider = gr.Slider(label="Audio Threshold", minimum=0, maximum=1, step=0.1, value=settings.get("audio_threshold", 0.7))
                        keywords_input = gr.Textbox(label="Keywords (comma-separated)", value=",".join(settings.get("text_keywords", [])))
                        image_size_dropdown = gr.Dropdown(label="Image Size", choices=["900p", "720p", "480p", "360p"], value=settings.get("image_size", "900p"))
                        use_gpu_checkbox = gr.Checkbox(label="Use GPU", value=settings.get("processor_settings", {}).get("use_gpu", True))
                        cpu_cores_slider = gr.Slider(label="CPU Cores", minimum=1, maximum=16, step=1, value=settings.get("processor_settings", {}).get("cpu_cores", 4))
                    with gr.Column():
                        reset_button = gr.Button("Reset to Default", variant="secondary")
                        save_button = gr.Button("Save Configuration", variant="primary")
                
                save_button.click(
                    fn=lambda x, y: save_persistent_settings({"input_path": x, "output_path": y}),
                    inputs=[input_folder, output_folder],
                    outputs=None
                )
                
                reset_button.click(
                    fn=lambda: save_persistent_settings({}),
                    inputs=None,
                    outputs=None
                )
                
        interface.launch(inbrowser=True)

if __name__ == "__main__":
    launch_gradio_interface()
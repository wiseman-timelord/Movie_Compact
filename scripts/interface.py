import os
import json
import gradio as gr
from data.temporary import SEARCH_CRITERIA
from scripts.generate import process_video
from scripts.utility import extract_frames, analyze_segment

def load_persistent_settings():
    if not os.path.exists("./data/persistent.json"):
        return {}
    with open("./data/persistent.json", "r") as f:
        return json.load(f)

def save_persistent_settings(settings):
    os.makedirs("./data", exist_ok=True)
    with open("./data/persistent.json", "w") as f:
        json.dump(settings, f, indent=4)

def update_temporary_settings():
    persistent_settings = load_persistent_settings()
    for key, value in persistent_settings.items():
        if key in SEARCH_CRITERIA:
            SEARCH_CRITERIA[key] = value

def update_keys(new_motion_threshold=None, new_texture_threshold=None, new_audio_threshold=None, new_keywords=None):
    persistent_settings = load_persistent_settings()
    if new_motion_threshold is not None:
        persistent_settings["motion_threshold"] = new_motion_threshold
        SEARCH_CRITERIA["motion_threshold"] = new_motion_threshold
    if new_texture_threshold is not None:
        persistent_settings["texture_threshold"] = new_texture_threshold
        SEARCH_CRITERIA["texture_threshold"] = new_texture_threshold
    if new_audio_threshold is not None:
        persistent_settings["audio_threshold"] = new_audio_threshold
        SEARCH_CRITERIA["audio_threshold"] = new_audio_threshold
    if new_keywords is not None:
        persistent_settings["text_keywords"] = new_keywords.split(",")
        SEARCH_CRITERIA["text_keywords"] = new_keywords.split(",")
    save_persistent_settings(persistent_settings)
    return "Settings updated successfully!"

def reset_settings():
    default_settings = {
        "motion_threshold": 0.5,
        "texture_threshold": 0.6,
        "audio_threshold": 0.7,
        "text_keywords": [],
    }
    save_persistent_settings(default_settings)
    update_temporary_settings()
    return "Settings reset to default."

def process_video_interface(video_path, motion_threshold, texture_threshold, audio_threshold, keywords):
    update_keys(motion_threshold, texture_threshold, audio_threshold, keywords)
    output_path = "./data/output_summary.mp4"
    process_video(video_path, output_path)
    return output_path

def launch_gradio_interface():
    update_temporary_settings()

    with gr.Blocks() as interface:
        with gr.Tabs():
            with gr.Tab("Consolidation"):
                gr.Markdown("# Video Summarization Interface")
                file_list_textbox = gr.Textbox(label="Files for Inclusion", interactive=False)
                with gr.Row():
                    browse_folders_btn = gr.Button("Browse Folders")
                    restart_session_btn = gr.Button("Restart Session")
                    exit_program_btn = gr.Button("Exit Program")
                video_input = gr.Video(label="Upload Video", format="mp4")
                process_btn = gr.Button("Process Video", variant="primary")
                video_output = gr.Video(label="Summary Video", format="mp4")

                process_btn.click(
                    fn=process_video_interface,
                    inputs=[video_input, motion_threshold_slider, texture_threshold_slider, audio_threshold_slider, keywords_input],
                    outputs=video_output,
                )

            with gr.Tab("Configuration"):
                gr.Markdown("# Configuration Settings")
                with gr.Row():
                    with gr.Column():
                        motion_threshold_slider = gr.Slider(
                            label="Motion Threshold",
                            minimum=0,
                            maximum=1,
                            step=0.1,
                            value=SEARCH_CRITERIA["motion_threshold"],
                        )
                        texture_threshold_slider = gr.Slider(
                            label="Texture Threshold",
                            minimum=0,
                            maximum=1,
                            step=0.1,
                            value=SEARCH_CRITERIA["texture_threshold"],
                        )
                        audio_threshold_slider = gr.Slider(
                            label="Audio Threshold",
                            minimum=0,
                            maximum=1,
                            step=0.1,
                            value=SEARCH_CRITERIA["audio_threshold"],
                        )
                        keywords_input = gr.Textbox(
                            label="Keywords (comma-separated)",
                            value=",".join(SEARCH_CRITERIA["text_keywords"]),
                        )
                    with gr.Column():
                        reset_btn = gr.Button("Reset to Default", variant="secondary")
                        save_btn = gr.Button("Save Configuration", variant="primary")

                save_btn.click(
                    fn=update_keys,
                    inputs=[motion_threshold_slider, texture_threshold_slider, audio_threshold_slider, keywords_input],
                    outputs=gr.Textbox(label="Status"),
                )
                reset_btn.click(
                    fn=reset_settings,
                    outputs=gr.Textbox(label="Status"),
                )

    interface.launch(inbrowser=True)

if __name__ == "__main__":
    launch_gradio_interface()
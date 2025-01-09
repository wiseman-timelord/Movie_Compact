# .\scripts\generate.py

import moviepy.editor as mp
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.utility import analyze_segment, extract_frames
from data.temporary import SEARCH_CRITERIA
import os
import moviepy.editor as mp
from utility import get_video_duration

def process_videos(input_paths, output_path, output_length, output_quality):
    summaries = []
    total_duration = sum(get_video_duration(path) for path in input_paths)
    for path in input_paths:
        summary = generate_summary(path, (output_length * get_video_duration(path)) / total_duration)
        summaries.append(summary)
    final_clip = mp.concatenate_videoclips(summaries)
    final_clip = final_clip.resize(output_quality['resolution'])
    final_clip.write_videofile(output_path, bitrate=output_quality['bitrate'])
    
def process_video(input_path, output_path):
    video_clip = mp.VideoFileClip(input_path)
    highlights = []

    frames = extract_frames(input_path)
    for i in range(1, len(frames)):
        if analyze_segment(frames[i - 1], frames[i]):
            start_time = i / video_clip.fps
            end_time = (i + 1) / video_clip.fps
            highlight = video_clip.subclip(start_time, end_time)
            highlights.append(highlight)

    if highlights:
        final_clip = mp.concatenate_videoclips(highlights)
        final_clip.write_videofile(output_path, codec='libx264')
    else:
        print("No highlights detected.")
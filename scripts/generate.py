import moviepy.editor as mp
from scripts.utility import analyze_segment, extract_frames
from data.temporary import SEARCH_CRITERIA

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
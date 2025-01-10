# .\scripts\generate.py

import moviepy.editor as mp
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.utility import analyze_segment, extract_frames
from data.temporary import SEARCH_CRITERIA
import os
import moviepy.editor as mp
from utility import get_video_duration, get_video_files

def process_videos(input_dir, output_dir):
    video_files = get_video_files(input_dir)
    if not video_files:
        print("No video files found in the input directory.")
        return
    for video_path in video_files:
        output_path = os.path.join(output_dir, os.path.basename(video_path))
        process_video(video_path, output_path)
        print(f"Processed {video_path} and saved summary to {output_path}")
    
def process_video(input_path, output_path):
    video_clip = mp.VideoFileClip(input_path)
    frames = extract_frames(input_path)
    highlights = []

    for i in range(1, len(frames)):
        if detect_motion_opencl(frames[i - 1], frames[i]):  # Use OpenCL for motion detection
            start_time = i / video_clip.fps
            end_time = (i + 1) / video_clip.fps
            highlight = video_clip.subclip(start_time, end_time)
            highlights.append(highlight)

    if highlights:
        final_clip = mp.concatenate_videoclips(highlights)
        final_clip.write_videofile(output_path, codec='libx264')
    else:
        print("No highlights detected.")
        
def detect_motion(frame1, frame2, threshold=SEARCH_CRITERIA['motion_threshold']):
    try:
        return detect_motion_opencl(frame1, frame2, threshold)
    except Exception as e:
        print(f"OpenCL motion detection failed: {e}. Falling back to CPU.")
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(gray1, gray2)
        _, diff = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        motion_score = np.mean(diff) / 255.0
        return motion_score > threshold
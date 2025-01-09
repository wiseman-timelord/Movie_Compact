# .\launcher.py

import os
import cv2
import numpy as np
import moviepy.editor as mp
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.video.compositing.concatenate import concatenate_videoclips
from data.temporary import SEARCH_CRITERIA

def detect_motion(frame1, frame2, threshold=SEARCH_CRITERIA['motion_threshold']):
    """Detect motion between two frames."""
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray1, gray2)
    _, diff = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    motion_score = np.mean(diff) / 255.0
    return motion_score > threshold

def detect_texture_change(frame1, frame2, threshold=SEARCH_CRITERIA['texture_threshold']):
    """Detect texture change between two frames."""
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    hist1 = cv2.calcHist([gray1], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([gray2], [0], None, [256], [0, 256])
    texture_score = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CHISQR)
    return texture_score > threshold

def analyze_segment(frame1, frame2):
    """Analyze segment for motion or texture changes."""
    motion = detect_motion(frame1, frame2)
    texture = detect_texture_change(frame1, frame2)
    return motion or texture

def extract_frames(video_path, frame_rate=30):
    """Extract frames from video at a specified frame rate."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

def process_video(input_path, output_path):
    """Process video and generate summary."""
    frames = extract_frames(input_path)
    highlights = []
    video_clip = VideoFileClip(input_path)
    for i in range(1, len(frames)):
        if analyze_segment(frames[i - 1], frames[i]):
            start_time = i / video_clip.fps
            end_time = (i + 1) / video_clip.fps
            highlight = video_clip.subclip(start_time, end_time)
            highlights.append(highlight)
    if highlights:
        final_clip = concatenate_videoclips(highlights)
        final_clip.write_videofile(output_path, codec='libx264')
    else:
        print("No highlights detected.")

def main():
    """Main function to launch the video summarization process."""
    input_path = "input_video.mp4"  # Replace with actual input path
    output_path = "output_summary.mp4"  # Replace with actual output path

    if not os.path.exists(input_path):
        print(f"Error: Input video file not found at {input_path}.")
        return

    print("Starting video summarization...")
    process_video(input_path, output_path)
    print(f"Video summary saved to {output_path}.")

if __name__ == "__main__":
    main()
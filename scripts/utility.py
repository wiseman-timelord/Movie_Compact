import cv2
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.temporary import SEARCH_CRITERIA
import moviepy.editor as mp

def get_video_duration(file_path):
    clip = mp.VideoFileClip(file_path)
    duration = clip.duration
    clip.close()
    return duration
    
def detect_motion(frame1, frame2, threshold=SEARCH_CRITERIA['motion_threshold']):
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray1, gray2)
    _, diff = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    motion_score = np.mean(diff) / 255.0
    return motion_score > threshold

def detect_texture_change(frame1, frame2, threshold=SEARCH_CRITERIA['texture_threshold']):
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    hist1 = cv2.calcHist([gray1], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([gray2], [0], None, [256], [0, 256])
    texture_score = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CHISQR)
    return texture_score > threshold

def analyze_segment(frame1, frame2):
    motion = detect_motion(frame1, frame2)
    texture = detect_texture_change(frame1, frame2)
    return motion or texture

def extract_frames(video_path, frame_rate=30):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames
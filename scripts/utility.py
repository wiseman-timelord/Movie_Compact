# .\scripts\utility.py

import pyopencl as cl
import cv2
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.temporary import SEARCH_CRITERIA
import moviepy.editor as mp

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

def get_video_files(directory):
    supported_extensions = ['.mp4', '.avi', '.mkv']
    video_files = [os.path.join(directory, f) for f in os.listdir(directory) if os.path.splitext(f)[1].lower() in supported_extensions]
    return sorted(video_files)

def get_video_duration(file_path):
    import moviepy.editor as mp
    clip = mp.VideoFileClip(file_path)
    duration = clip.duration
    clip.close()
    return duration
    
load_hardware_config

def detect_motion_avx2(frame1, frame2, threshold=SEARCH_CRITERIA['motion_threshold']):
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray1, gray2)
    motion_score = np.mean(diff) / 255.0
    return motion_score > threshold

def detect_motion(frame1, frame2, threshold=SEARCH_CRITERIA['motion_threshold']):
    hardware_config = load_hardware_config()
    if hardware_config["OpenCL"]:
        # OpenCL implementation
        pass
    elif hardware_config["Avx2"]:
        return detect_motion_avx2(frame1, frame2, threshold)
    else:
        # Fallback to x64 implementation
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(gray1, gray2)
        _, diff = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        motion_score = np.mean(diff) / 255.0
        return motion_score > threshold

def detect_motion_opencl(frame1, frame2, threshold=SEARCH_CRITERIA['motion_threshold']):
    # Initialize OpenCL context and queue
    platforms = cl.get_platforms()
    gpu_devices = platforms[0].get_devices(device_type=cl.device_type.GPU)
    context = cl.Context(devices=gpu_devices)
    queue = cl.CommandQueue(context)

    # Convert frames to grayscale
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Create OpenCL buffers
    mf = cl.mem_flags
    gray1_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=gray1)
    gray2_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=gray2)
    diff_buf = cl.Buffer(context, mf.WRITE_ONLY, gray1.nbytes)

    # Build and execute OpenCL kernel
    kernel_code = """
    __kernel void detect_motion(__global const uchar* gray1, __global const uchar* gray2, __global uchar* diff) {
        int id = get_global_id(0);
        diff[id] = abs(gray1[id] - gray2[id]);
    }
    """
    program = cl.Program(context, kernel_code).build()
    program.detect_motion(queue, gray1.shape, None, gray1_buf, gray2_buf, diff_buf)

    # Read back the result
    diff = np.empty_like(gray1)
    cl.enqueue_copy(queue, diff, diff_buf)

    # Calculate motion score
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
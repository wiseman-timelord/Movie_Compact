# .\installer.py

import os
import platform
import cpuinfo
import subprocess
import sys
import json
import datetime

def log_event(message):
    """Log events to events.txt with timestamp."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    event_file = os.path.join("data", "events.txt")
    with open(event_file, "a") as f:
        f.write(f"[{timestamp}] {message}\n")

def detect_hardware():
    """Detect system hardware capabilities and save to hardware.txt."""
    print("Detecting hardware capabilities...")
    hardware_info = {
        "x64": platform.machine().endswith("64"),
        "Avx2": False,
        "Aocl": False,
        "OpenCL": False,
    }

    try:
        cpu_info = cpuinfo.get_cpu_info()
        hardware_info["Avx2"] = "avx2" in cpu_info.get("flags", [])
    except Exception as e:
        log_event(f"Error detecting AVX2: {e}")
        print(f"Error detecting AVX2: {e}")

    try:
        import pyopencl as cl
        platforms = cl.get_platforms()
        hardware_info["OpenCL"] = len(platforms) > 0
        if hardware_info["OpenCL"]:
            hardware_info["Aocl"] = any("AMD" in p.name for p in platforms)
    except Exception as e:
        log_event(f"Error detecting OpenCL: {e}")
        print(f"Error detecting OpenCL: {e}")

    hardware_file = os.path.join("data", "hardware.txt")
    with open(hardware_file, "w") as f:
        for key, value in hardware_info.items():
            f.write(f"{key}: {value}\n")
    log_event("Hardware configuration saved to hardware.txt")
    print("Hardware configuration saved to hardware.txt")
    return hardware_info

def ensure_directories():
    """Create all required directories if they don't exist."""
    print("Creating required directories...")
    directories = ["data", "input", "output", "work"]
    for directory in directories:
        path = os.path.join(".", directory)
        if not os.path.exists(path):
            os.makedirs(path)
            log_event(f"Created directory: {directory}")
    print("All required directories created.")

def create_temporary_py():
    """Create temporary.py with SEARCH_CRITERIA configuration."""
    print("Creating temporary.py with search criteria...")
    temp_file = os.path.join("data", "temporary.py")
    content = '''# Search criteria configuration for video analysis
SEARCH_CRITERIA = {
    # Motion detection threshold (0.0 to 1.0)
    # Used by launcher.py and utility.py for frame comparison
    'motion_threshold': 0.5,
    
    # Texture change threshold (0.0 to 1.0)
    # Used by utility.py for texture analysis
    'texture_threshold': 0.6,
    
    # Audio level threshold (0.0 to 1.0)
    # Used for audio peak detection
    'audio_threshold': 0.7,
    
    # Frame processing settings
    'frame_settings': {
        'sample_rate': 30,  # Frames per second to analyze
        'min_segment': 2,   # Minimum segment length in seconds
        'max_segment': 30,  # Maximum segment length in seconds
        'batch_size': 100   # Number of frames to process in one batch
    },
    
    # Video processing settings
    'video_settings': {
        'target_fps': 30,           # Target FPS for output
        'min_clip_length': 2,       # Minimum clip length in seconds
        'resolution_height': 720,    # Target resolution height
        'codec': 'libx264',         # Default video codec
        'audio_codec': 'aac'        # Default audio codec
    }
}

# Video processing configuration
VIDEO_CONFIG = {
    'supported_formats': ['.mp4', '.avi', '.mkv'],
    'temp_directory': 'work',
    'output_directory': 'output',
    'input_directory': 'input'
}

# Hardware optimization settings
# Used by utility.py and generate.py for hardware acceleration
HARDWARE_CONFIG = {
    'use_gpu': True,
    'use_opencl': True,
    'use_avx2': True,
    'gpu_batch_size': 32,
    'cpu_threads': 4,
    'opencl_platform_preference': ['NVIDIA', 'AMD', 'Intel']
}
'''
    with open(temp_file, "w") as f:
        f.write(content)
    log_event("Created temporary.py with default search criteria")
    print("Created temporary.py with default search criteria.")

def create_persistent_json():
    """Create persistent.json with default settings."""
    print("Creating persistent.json with default settings...")
    default_settings = {
        # Basic thresholds
        "motion_threshold": 0.5,
        "texture_threshold": 0.6,
        "audio_threshold": 0.7,

        # Frame analysis settings
        "frame_settings": {
            "sample_rate": 30,
            "min_segment": 2,
            "max_segment": 30,
            "batch_size": 100
        },

        # Video configuration
        "video_settings": {
            "target_fps": 30,
            "min_clip_length": 2,
            "target_length": 30,  # minutes
            "resolution_height": 720,
            "codec": "libx264",
            "audio_codec": "aac",
            "preview_height": 360  # height for preview processing
        },

        # Processing settings
        "processor_settings": {
            "use_gpu": True,
            "cpu_cores": 4,
            "opencl_enabled": True,
            "avx2_enabled": True,
            "gpu_batch_size": 32
        },

        # File paths
        "paths": {
            "input_path": "input",
            "output_path": "output",
            "work_path": "work"
        },

        # Scene detection
        "scene_settings": {
            "min_scene_length": 2,    # seconds
            "max_scene_length": 300,   # seconds
            "scene_threshold": 30.0,   # threshold for scene change detection
            "action_threshold": 0.3    # threshold for action sequence detection
        },

        # Speed adjustment
        "speed_settings": {
            "max_speed_factor": 4.0,   # maximum speedup for non-action scenes
            "min_speed_factor": 1.0,   # minimum speedup (normal speed)
            "transition_frames": 30     # frames for speed transition
        }
    }

    persistent_file = os.path.join("data", "persistent.json")
    with open(persistent_file, "w") as f:
        json.dump(default_settings, f, indent=4)
    log_event("Created persistent.json with default settings")
    print("Created persistent.json with default settings.")


def create_requirements_file():
    """Create requirements.txt with necessary packages."""
    print("Creating requirements.txt...")
    requirements = [
        "moviepy==1.0.3",
        "numpy==1.26.0",
        "pandas==2.1.3",
        "psutil==6.1.1",
        "gradio==5.9.1",
        "opencv-python==4.8.1.78",
        "pyopencl==2023.1.1",
        "librosa==0.10.1",
        "pydub==0.25.1",
        "py-cpuinfo==9.0.0",
        "scikit-image==0.22.0",  # Added for additional image processing
        "scipy==1.11.4"          # Required by other packages
    ]
    req_file = os.path.join("data", "requirements.txt")
    with open(req_file, "w") as f:
        for package in requirements:
            f.write(f"{package}\n")
    log_event("Created requirements.txt")
    print("Created requirements.txt in the data directory.")

def verify_installation():
    """Verify all required files and directories exist."""
    print("\nVerifying installation...")
    required_dirs = {
        ".": ["data", "input", "output", "work", "scripts"]  # Added scripts
    }
    required_files = {
        "data": ["persistent.json", "requirements.txt", "hardware.txt", "events.txt"]  # Removed temporary.py
    }
    
    status = True
    
    # Check directories
    for base_path, dirs in required_dirs.items():
        for directory in dirs:
            dir_path = os.path.join(base_path, directory)
            if not os.path.exists(dir_path):
                print(f"ERROR: Directory {dir_path} is missing!")
                status = False
    
    # Check files
    for base_path, files in required_files.items():
        for file in files:
            file_path = os.path.join(base_path, file)
            if not os.path.exists(file_path):
                print(f"ERROR: File {file_path} is missing!")
                status = False
    
    if status:
        print("Installation verification successful!")
        log_event("Installation verification completed successfully")
    else:
        print("Installation verification failed!")
        log_event("Installation verification failed")
    return status

def install_requirements():
    """Install required packages from requirements.txt."""
    print("\nInstalling requirements...")
    req_file = os.path.join("data", "requirements.txt")
    try:
        # Upgrade pip first
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        # Install requirements
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", req_file])
        log_event("Requirements installed successfully")
        print("Requirements installed successfully.")
        return True
    except subprocess.CalledProcessError as e:
        error_msg = f"Error: Failed to install requirements. {e}"
        log_event(error_msg)
        print(error_msg)
        return False

def main():
    print("Starting installation process...")
    log_event("Starting installation process")
    
    try:
        # Create directories first
        ensure_directories()
        
        # Detect hardware and create configuration files
        hardware_info = detect_hardware()
        # Removed create_temporary_py() call
        create_persistent_json()
        create_requirements_file()
        
        # Install requirements
        if not install_requirements():
            print("Installation failed due to requirement installation error.")
            log_event("Installation failed - requirement installation error")
            return False
        
        # Verify installation
        if not verify_installation():
            print("Installation failed verification checks.")
            log_event("Installation failed - verification checks")
            return False
        
        print("\nInstallation complete!")
        log_event("Installation completed successfully")
        return True
        
    except Exception as e:
        error_msg = f"Installation failed with error: {str(e)}"
        print(error_msg)
        log_event(error_msg)
        return False

if __name__ == "__main__":
    main()
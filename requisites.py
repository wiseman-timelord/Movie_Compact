# .\requisites.py

import os
import platform
import subprocess
import sys
import json
import datetime

# Define the base directory as the location of this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def log_event(message):
    """Log events to events.txt with timestamp."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    event_file = os.path.join(BASE_DIR, "data", "events.txt")
    with open(event_file, "a") as f:
        f.write(f"[{timestamp}] {message}\n")

def detect_hardware():
    """Detect system hardware capabilities and save to hardware.json."""
    print("Detecting hardware capabilities...")
    hardware_info = {
        "x64": platform.machine().endswith("64"),
        "AVX2": False,  # Corrected key case to match convention
        "AOCL": False,  # Corrected key case
        "OpenCL": False,
    }

    try:
        import cpuinfo
        cpu_info = cpuinfo.get_cpu_info()
        hardware_info["AVX2"] = "avx2" in cpu_info.get("flags", [])
    except Exception as e:
        log_event(f"Error detecting AVX2: {e}")
        print(f"Error detecting AVX2: {e}")

    try:
        import pyopencl as cl
        platforms = cl.get_platforms()
        hardware_info["OpenCL"] = len(platforms) > 0
        if hardware_info["OpenCL"]:
            hardware_info["AOCL"] = any("AMD" in p.name for p in platforms)
    except Exception as e:
        log_event(f"Error detecting OpenCL: {e}")
        print(f"Error detecting OpenCL: {e}")

    hardware_file = os.path.join(BASE_DIR, "data", "hardware.json")
    with open(hardware_file, "w") as f:
        json.dump(hardware_info, f, indent=4)
    log_event("Hardware configuration saved to hardware.json")
    print("Hardware configuration saved to hardware.json")
    return hardware_info

def ensure_directories():
    """Create all required directories if they don't exist."""
    print("Creating required directories...")
    directories = ["data", "input", "output", "work"]
    for directory in directories:
        path = os.path.join(BASE_DIR, directory)
        if not os.path.exists(path):
            os.makedirs(path)
            log_event(f"Created directory: {directory}")
    print("All required directories created.")

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
            "target_length": 30,
            "resolution_height": 720,
            "codec": "libx264",
            "audio_codec": "aac",
            "preview_height": 360
        },

        # Processing settings (existing)
        "processor_settings": {
            "use_gpu": True,
            "cpu_cores": 4,
            "opencl_enabled": True,
            "avx2_enabled": True,
            "gpu_batch_size": 32
        },

        # Add hardware preferences
        "hardware_preferences": {
            "use_opencl": True,
            "use_avx2": True,
            "use_aocl": True
        },

        # File paths
        "paths": {
            "input_path": "input",
            "output_path": "output",
            "work_path": "work"
        },

        # Scene detection
        "scene_settings": {
            "min_scene_length": 2,
            "max_scene_length": 300,
            "scene_threshold": 30.0,
            "action_threshold": 0.3
        },

        # Speed adjustment
        "speed_settings": {
            "max_speed_factor": 4.0,
            "min_speed_factor": 1.0,
            "transition_frames": 30
        }
    }

    persistent_file = os.path.join(BASE_DIR, "data", "persistent.json")
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
        "scikit-image==0.22.0",
        "scipy==1.11.4"
    ]
    req_file = os.path.join(BASE_DIR, "data", "requirements.txt")
    with open(req_file, "w") as f:
        for package in requirements:
            f.write(f"{package}\n")
    log_event("Created requirements.txt")
    print("Created requirements.txt in the data directory.")

def verify_installation():
    """Verify all required files and directories exist."""
    print("\nVerifying installation...")
    required_dirs = [os.path.join(BASE_DIR, d) for d in ["data", "input", "output", "work", "scripts"]]
    required_files = [os.path.join(BASE_DIR, "data", f) for f in ["persistent.json", "requirements.txt", "hardware.json", "events.txt"]]
    
    status = True
    
    # Check directories
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            print(f"ERROR: Directory {dir_path} is missing!")
            status = False
    
    # Check files
    for file_path in required_files:
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

def main():
    """Main function to orchestrate the installation process."""
    print("Starting installation process...")
    log_event("Installation process started")
    
    try:
        # Check if running in a virtual environment
        if sys.prefix == sys.base_prefix:
            print("Warning: Not running in a virtual environment. Dependencies will be installed globally.")
            log_event("Warning: Not running in a virtual environment.")
        
        # Create directories first
        ensure_directories()
        
        # Create requirements file and install dependencies
        create_requirements_file()
        if not install_requirements():
            print("Installation failed due to requirement installation error.")
            log_event("Installation failed - requirement installation error")
            return False
        
        # Detect hardware after dependencies are installed
        hardware_info = detect_hardware()
        
        # Create persistent configuration
        create_persistent_json()
        
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
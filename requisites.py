import os
import shutil
import subprocess
import sys
import json
import time
import platform

# Define the base directory as the location of this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def ensure_directories():
    """Create all required directories if they don't exist."""
    print("Creating required directories...")
    directories = ["data", "input", "output", "work"]
    for directory in directories:
        path = os.path.join(BASE_DIR, directory)
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"Created directory: {directory}")
    print("All required directories created.")

def create_init_py():
    """Create essential initialization files."""
    print("\nCreating essential files...")
    
    # Create scripts/__init__.py
    scripts_init = os.path.join(BASE_DIR, "scripts", "__init__.py")
    if not os.path.exists(scripts_init):
        with open(scripts_init, "w") as f:
            f.write("# Package initialization file\n")
        print("Created scripts/__init__.py")
    
    print("Essential files created.")

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
        "scipy==1.11.4",
        "siphash24==1.4",
        "moviepy==1.0.3"
    ]
    req_file = os.path.join(BASE_DIR, "data", "requirements.txt")
    with open(req_file, "w") as f:
        for package in requirements:
            f.write(f"{package}\n")
    print("Created requirements.txt in the data directory.")

def detect_hardware():
    """Detect system hardware capabilities and save to hardware.json."""
    print("Detecting hardware capabilities...")
    hardware_info = {
        "x64": platform.machine().endswith("64"),
        "AVX2": False,
        "AOCL": False,
        "OpenCL": False,
    }

    try:
        import cpuinfo
        cpu_info = cpuinfo.get_cpu_info()
        hardware_info["AVX2"] = "avx2" in cpu_info.get("flags", [])
    except Exception as e:
        print(f"Error detecting AVX2: {e}")
        time.sleep(5)  # Pause for 5 seconds after error

    try:
        import pyopencl as cl
        platforms = cl.get_platforms()
        hardware_info["OpenCL"] = len(platforms) > 0
        if hardware_info["OpenCL"]:
            hardware_info["AOCL"] = any("AMD" in p.name for p in platforms)
    except Exception as e:
        print(f"Error detecting OpenCL: {e}")
        time.sleep(5)  # Pause for 5 seconds after error

    hardware_file = os.path.join(BASE_DIR, "data", "hardware.json")
    with open(hardware_file, "w") as f:
        json.dump(hardware_info, f, indent=4)
    print("Hardware configuration saved to hardware.json")
    return hardware_info

def create_persistent_json():
    """Create persistent.json with default settings."""
    print("Creating persistent.json with default settings...")
    default_settings = {
        "motion_threshold": 0.5,
        "texture_threshold": 0.6,
        "audio_threshold": 0.7,
        "frame_settings": {
            "sample_rate": 30,
            "min_segment": 2,
            "max_segment": 30,
            "batch_size": 100
        },
        "video_settings": {
            "target_fps": 30,
            "min_clip_length": 2,
            "target_length": 30,
            "resolution_height": 720,
            "codec": "libx264",
            "audio_codec": "aac",
            "preview_height": 360
        },
        "processor_settings": {
            "use_gpu": True,
            "cpu_cores": 4,
            "opencl_enabled": True,
            "avx2_enabled": True,
            "gpu_batch_size": 32
        },
        "hardware_preferences": {
            "use_opencl": True,
            "use_avx2": True,
            "use_aocl": True
        },
        "paths": {
            "input_path": "input",
            "output_path": "output",
            "work_path": "work"
        },
        "scene_settings": {
            "min_scene_length": 2,
            "max_scene_length": 300,
            "scene_threshold": 30.0,
            "action_threshold": 0.3
        },
        "speed_settings": {
            "max_speed_factor": 4.0,
            "min_speed_factor": 1.0,
            "transition_frames": 30
        }
    }

    persistent_file = os.path.join(BASE_DIR, "data", "persistent.json")
    with open(persistent_file, "w") as f:
        json.dump(default_settings, f, indent=4)
    print("Created persistent.json with default settings.")

def verify_installation():
    """Verify all required files and directories exist."""
    print("\nVerifying installation...")
    required_dirs = [os.path.join(BASE_DIR, d) for d in ["data", "input", "output", "work"]]
    required_files = [os.path.join(BASE_DIR, "data", f) for f in ["persistent.json", "requirements.txt", "hardware.json"]]
    
    status = True
    
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            print(f"ERROR: Directory {dir_path} is missing!")
            status = False
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"ERROR: File {file_path} is missing!")
            status = False
    
    if status:
        print("Installation verification successful!")
    else:
        print("Installation verification failed!")
        time.sleep(5)  # Pause for 5 seconds if verification fails
    return status

def main():
    """Main function to orchestrate the installation process."""
    print("Starting installation process...")
    
    try:
        # Create directories first
        ensure_directories()
       
        # Create files
        create_init_py()
        create_requirements_file()
        
        # Check if running in virtual environment
        if sys.prefix == sys.base_prefix:
            print("Not running in virtual environment. Setting up virtual environment...")
            venv_path = os.path.join(BASE_DIR, "venv")
            if os.path.exists(venv_path):
                print("Removing existing virtual environment...")
                shutil.rmtree(venv_path)
            print("Creating new virtual environment...")
            subprocess.check_call([sys.executable, "-m", "venv", venv_path])
            venv_python = os.path.join(venv_path, "Scripts", "python.exe") if sys.platform == "win32" else os.path.join(venv_path, "bin", "python")
            req_file = os.path.join(BASE_DIR, "data", "requirements.txt")
            print("Installing requirements into virtual environment...")
            subprocess.check_call([venv_python, "-m", "pip", "install", "-r", req_file])
            print("Restarting script with virtual environment...")
            subprocess.check_call([venv_python, __file__])
            sys.exit(0)
        
        # If we reach here, we are running in the virtual environment
        print("Running in virtual environment.")
        
        # Detect hardware
        hardware_info = detect_hardware()
        
        # Create persistent configuration
        create_persistent_json()
        
        # Verify installation
        if not verify_installation():
            print("Installation failed verification checks.")
            time.sleep(5)  # Pause for 5 seconds after error
            return False
        
        print("\nInstallation complete!")
        print("To run the application, activate the virtual environment with 'venv\\Scripts\\activate' and then run 'python launcher.py'.")
        return True
        
    except Exception as e:
        error_msg = f"Installation failed with error: {str(e)}"
        print(error_msg)
        time.sleep(5)  # Pause for 5 seconds after error
        return False

if __name__ == "__main__":
    main()
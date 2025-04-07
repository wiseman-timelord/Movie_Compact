# Script: `.\requisites.py`

# Imports
import os
import shutil
import subprocess
import sys
import json
import time
import platform

# Globals
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Functions...
def clean_previous_installation():
    """Remove existing installation artifacts from previous runs."""
    print("Cleaning previous installation...")
    targets = ["data"]
    # Only clean 'venv' if not in a virtual environment
    if sys.prefix == sys.base_prefix:
        targets.append("venv")
    
    for target in targets:
        target_path = os.path.join(BASE_DIR, target)
        if os.path.exists(target_path):
            try:
                if os.path.isdir(target_path):
                    shutil.rmtree(target_path)
                    print(f"Removed existing {target} directory")
                    time.sleep(1)
                else:
                    os.remove(target_path)
                    print(f"Removed existing {target} file")
                    time.sleep(1)
            except Exception as e:
                print(f"Error removing {target}: {str(e)}")
                time.sleep(5)
                raise

def ensure_directories():
    """Create all required directories if they don't exist."""
    print("Creating required directories...")
    directories = ["data", "data/temp", "input", "output", "scripts"]
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
    """Detect system hardware capabilities (no longer saves to hardware.json)."""
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
        time.sleep(5)

    try:
        import pyopencl as cl
        platforms = cl.get_platforms()
        hardware_info["OpenCL"] = len(platforms) > 0
        if hardware_info["OpenCL"]:
            hardware_info["AOCL"] = any("AMD" in p.name for p in platforms)
    except Exception as e:
        print(f"Error detecting OpenCL: {e}")
        time.sleep(5)

    return hardware_info  # Return data instead of saving to file

def create_persistent_json(hardware_info):
    """Create persistent.json with unified configuration."""
    print("Creating persistent.json with unified configuration...")
    
    default_settings = {
        # Hardware Configuration (detected capabilities)
        "hardware_config": hardware_info,
        
        # Processing Configuration
        "processing_config": {
            # Video Settings
            "preview_height": 360,
            "target_height": 720,
            "frame_sample_rate": 30,
            "supported_formats": [".mp4", ".avi", ".mkv"],
            "output_codec": "libx264",
            "audio_codec": "aac",
            
            # Hardware Acceleration Preferences
            "hardware_acceleration": {
                "use_gpu": True,
                "use_opencl": hardware_info.get("OpenCL", False),
                "use_avx2": hardware_info.get("AVX2", False),
                "opencl_platform_preference": ["NVIDIA", "AMD", "Intel"]
            },
            
            # Performance Settings
            "performance": {
                "gpu_batch_size": 32,
                "cpu_threads": 4,
                "frame_buffer_size": 30
            },
            
            # Scene Detection
            "static_threshold": 0.98,
            "menu_threshold": 0.85,
            "motion_threshold": 0.3,
            "texture_threshold": 0.4,
            "min_scene_duration": 2.0,
            "max_scene_duration": 300.0,
            
            # Speed Adjustment
            "speed_settings": {
                "max_speed_factor": 4.0,
                "min_speed_factor": 1.0,
                "transition_frames": 30
            }
        },
        
        # Audio Configuration
        "audio_config": {
            "sample_rate": 44100,
            "window_size": 2048,
            "hop_length": 512,
            "preserve_pitch": True,
            "enhance_audio": True
        },
        
        # Path Configuration
        "path_config": {
            "input_path": "input",
            "output_path": "output",
            "work_path": "data/temp"
        }
    }

    persistent_file = os.path.join(BASE_DIR, "data", "persistent.json")
    with open(persistent_file, "w") as f:
        json.dump(default_settings, f, indent=4)
    print("Created unified persistent.json configuration.")

def verify_installation():
    """Verify installation with new config structure."""
    print("\nVerifying installation...")
    required = [
        os.path.join(BASE_DIR, "data", "persistent.json"),
        os.path.join(BASE_DIR, "data", "requirements.txt"),
        os.path.join(BASE_DIR, "scripts", "__init__.py")
    ]
    
    # Check directory structure
    dirs = ["data", "data/temp", "input", "output", "scripts"]
    missing_dirs = [d for d in dirs if not os.path.exists(os.path.join(BASE_DIR, d))]
    
    # Check required files
    missing_files = [f for f in required if not os.path.exists(f)]
    
    if not missing_dirs and not missing_files:
        print("Installation verified successfully!")
        return True
    
    if missing_dirs:
        print("Missing directories:")
        for d in missing_dirs: print(f" - {d}")
    
    if missing_files:
        print("Missing files:")
        for f in missing_files: print(f" - {f}")
    
    print("Verification failed!")
    time.sleep(5)
    return False

def main():
    """Main function to orchestrate the installation process."""
    print("Starting installation process...")
    
    try:
        in_venv = sys.prefix != sys.base_prefix
        
        if not in_venv:
            # Initial setup (outside venv)
            clean_previous_installation()
            ensure_directories()
            create_init_py()
            create_requirements_file()
            
            # Setup virtual environment
            print("Not running in virtual environment. Setting up virtual environment...")
            venv_path = os.path.join(BASE_DIR, "venv")
            
            # Remove residual venv if exists
            if os.path.exists(venv_path):
                print("Removing residual virtual environment...")
                shutil.rmtree(venv_path)
                time.sleep(1)
            
            # Create new venv
            print("Creating new virtual environment...")
            subprocess.check_call([sys.executable, "-m", "venv", venv_path])
            venv_python = os.path.join(venv_path, "Scripts", "python.exe") if sys.platform == "win32" else os.path.join(venv_path, "bin", "python")
            req_file = os.path.join(BASE_DIR, "data", "requirements.txt")
            
            # Upgrade pip first
            print("Upgrading pip...")
            subprocess.check_call([venv_python, "-m", "pip", "install", "--upgrade", "pip"])
            
            # Install wheel and requirements
            print("Installing wheel package...")
            subprocess.check_call([venv_python, "-m", "pip", "install", "wheel"])
            print("Installing requirements into virtual environment...")
            subprocess.check_call([venv_python, "-m", "pip", "install", "--use-pep517", "-r", req_file])
            
            # Restart script in venv
            print("Restarting script with virtual environment...")
            subprocess.check_call([venv_python, __file__])
            sys.exit(0)
        else:
            # Running in venv: configuration steps
            print("Running in virtual environment.")
            
            # Ensure directories exist (no cleanup)
            ensure_directories()
            
            # Detect hardware and create config
            hardware_info = detect_hardware()
            create_persistent_json(hardware_info)
            
            # Verify installation
            if not verify_installation():
                print("Installation failed verification checks.")
                time.sleep(3)
                return False
            
            print("\nInstallation complete!")
            print("To run the application, activate the virtual environment with 'venv\\Scripts\\activate' (Windows) or 'source venv/bin/activate' (Unix), then run 'python launcher.py'.")
            return True
            
    except Exception as e:
        error_msg = f"Installation failed with error: {str(e)}"
        print(error_msg)
        time.sleep(5)
        return False

if __name__ == "__main__":
    main()
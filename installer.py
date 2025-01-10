import os
import platform
import cpuinfo
import subprocess
import sys
import json

def detect_hardware():
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
        print(f"Error detecting AVX2: {e}")

    try:
        import pyopencl as cl
        platforms = cl.get_platforms()
        hardware_info["OpenCL"] = len(platforms) > 0
    except Exception as e:
        print(f"Error detecting OpenCL: {e}")

    hardware_file = os.path.join("data", "hardware.txt")
    os.makedirs("data", exist_ok=True)
    with open(hardware_file, "w") as f:
        for key, value in hardware_info.items():
            f.write(f"{key}: {value}\n")
    print("Hardware configuration saved to hardware.txt.")

def ensure_data_directory():
    data_dir = os.path.join(".", "data")
    input_dir = os.path.join(".", "input")
    output_dir = os.path.join(".", "output")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    if not os.path.exists(input_dir):
        os.makedirs(input_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print("Data, input, and output directories created.")


def create_persistent_json():
    default_settings = {
        "motion_threshold": 0.5,
        "texture_threshold": 0.6,
        "audio_threshold": 0.7,
        "text_keywords": [],
        "image_size": "900p",
        "target_length": 5,
        "output_quality": "720p",
        "processor_settings": {
            "use_gpu": True,
            "cpu_cores": 4,
        },
    }
    persistent_file = os.path.join("data", "persistent.json")
    if not os.path.exists(persistent_file):
        with open(persistent_file, "w") as f:
            json.dump(default_settings, f, indent=4)
        print("Created persistent.json with default settings.")

def create_requirements_file():
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
    ]
    req_file = os.path.join("data", "requirements.txt")
    if not os.path.exists(req_file):
        with open(req_file, "w") as f:
            for package in requirements:
                f.write(f"{package}\n")
        print("Created requirements.txt in the data directory.")

def install_requirements():
    req_file = os.path.join("data", "requirements.txt")
    print("Installing requirements...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", req_file])
        print("Requirements installed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error: Failed to install requirements. {e}")
        sys.exit(1)

def main():
    print("Starting installation process...")
    ensure_data_directory()
    detect_hardware()
    create_persistent_json()
    create_requirements_file()
    install_requirements()
    print("Installation complete.")

if __name__ == "__main__":
    main()
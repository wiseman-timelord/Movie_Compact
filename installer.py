import os
import subprocess
import sys
import json

def ensure_data_directory():
    data_dir = os.path.join(".", "data")
    if not os.path.exists(data_dir):
        print("Creating data directory...")
        os.makedirs(data_dir)
        print("Data directory created.")

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
        # Other settings as needed
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
    ]
    req_file = os.path.join("data", "requirements.txt")
    if not os.path.exists(req_file):
        with open(req_file, "w") as f:
            for package in requirements:
                f.write(f"{package}\n")
        print("Created requirements.txt in the data directory.")

def create_temporary_py():
    persistent_file = os.path.join("data", "persistent.json")
    if os.path.exists(persistent_file):
        with open(persistent_file, "r") as f:
            settings = json.load(f)
        search_criteria = {
            "motion_threshold": settings.get("motion_threshold", 0.5),
            "texture_threshold": settings.get("texture_threshold", 0.6),
            # Add other criteria as needed
        }
        temporary_file = os.path.join("data", "temporary.py")
        with open(temporary_file, "w") as f:
            f.write(f"SEARCH_CRITERIA = {search_criteria}")
        print("Created temporary.py with search criteria.")

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
    create_persistent_json()
    create_requirements_file()
    create_temporary_py()
    install_requirements()
    print("Installation complete.")

if __name__ == "__main__":
    main()
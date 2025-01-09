import os
import subprocess
import sys
import json

def create_requirements_file():
    requirements = [
        "moviepy==1.0.3",
        "numpy==1.26.0",
        "pandas==2.1.3",
        "psutil==6.1.1",
        "gradio==5.9.1",
        "opencv-python==4.8.1.78",
    ]
    with open("data/requirements.txt", "w") as f:
        for package in requirements:
            f.write(f"{package}\n")
    print("Created requirements.txt in the data directory.")

def install_requirements():
    if not os.path.exists("data/requirements.txt"):
        print("Error: requirements.txt not found. Creating it now...")
        create_requirements_file()

    print("Installing requirements...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "data/requirements.txt"])
        print("Requirements installed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error: Failed to install requirements. {e}")
        sys.exit(1)

def ensure_data_directory():
    if not os.path.exists("data"):
        print("Creating data directory...")
        os.makedirs("data")
        print("Data directory created.")

def create_persistent_json():
    default_settings = {
        "motion_threshold": 0.5,
        "texture_threshold": 0.6,
        "audio_threshold": 0.7,
        "text_keywords": [],
    }
    with open("data/persistent.json", "w") as f:
        json.dump(default_settings, f, indent=4)
    print("Created persistent.json with default settings.")

def main():
    print("Starting installation process...")
    ensure_data_directory()
    create_requirements_file()
    install_requirements()
    create_persistent_json()
    print("Installation complete.")

if __name__ == "__main__":
    main()
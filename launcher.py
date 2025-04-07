# launcher.py

# Imports...
print("Initializing Imports..")
import os, sys, json, time, traceback
from moviepy.editor import VideoFileClip as mp_VideoFileClip 
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from scripts.utility import (
    load_settings,
    cleanup_work_directory,
    AudioAnalyzer,
    SceneManager,
    PreviewGenerator,
    MemoryManager,
    ErrorHandler,
    CoreUtilities
)
from scripts.process import VideoProcessor
from scripts.interface import launch_gradio_interface
from scripts.analyze import VideoAnalyzer
from scripts.temporary import SCENE_CONFIG
print("..Imports Initialized.")

# Classes...
class MovieCompact:
    def __init__(self):
        # Import configs dynamically to avoid circular imports
        from scripts.temporary import PROCESSING_CONFIG, MEMORY_CONFIG, SCENE_CONFIG
        # CoreUtilities, ErrorHandler, and other classes are assumed to be imported elsewhere or defined
        self.core = CoreUtilities()  # Utility class for core operations
        self.settings = load_settings()  # Load settings from a settings file or default
        self.hardware_config = self.settings.get('hardware_config', {})  # Hardware settings
        self.processing_config = PROCESSING_CONFIG  # Processing parameters
        self.memory_config = MEMORY_CONFIG  # Memory management settings
        self.error_handler = ErrorHandler()  # Initialize error handler for logging/reporting errors
        # Define required directories and files for the project structure
        self.required_dirs = ["data", "data/temp", "input", "output"]
        self.required_files = [
            os.path.join("data", "persistent.json"),
            os.path.join("data", "requirements.txt"),
            os.path.join("scripts", "__init__.py")
        ]
        self.validate_environment()  # Ensure environment is set up correctly
        # Initialize processing components
        self.analyzer = VideoAnalyzer(settings=self.settings)
        self.processor = VideoProcessor(settings=self.settings, analyzer=self.analyzer)
        self.audio_analyzer = AudioAnalyzer()
        self.scene_manager = SceneManager(scene_config=SCENE_CONFIG)
        self.preview_generator = PreviewGenerator()

    def validate_environment(self) -> None:
        """Ensure required directories and files exist."""
        # Create directories
        for path in self.required_dirs:
            os.makedirs(path, exist_ok=True)
            if not os.path.exists(path):
                print(f"Created directory: {path}")
                time.sleep(0.5)

        # Verify/Create files
        for file_path in self.required_files:
            if not os.path.exists(file_path):
                if "persistent.json" in file_path:
                    self._create_default_settings(file_path)
                elif "__init__.py" in file_path:
                    open(file_path, 'w').close()
                else:
                    open(file_path, 'w').close()
                print(f"Created file: {file_path}")
                time.sleep(0.5)

    def _create_default_settings(self, path: str) -> None:
        """Create default persistent.json with hardware config."""
        from scripts.temporary import ANALYSIS_CONFIG  # Add missing import
        default_settings = {
            "hardware_config": self.hardware_config,
            "processing_config": self.processing_config,
            "analysis_config": ANALYSIS_CONFIG
        }
        with open(path, 'w') as f:
            json.dump(default_settings, f, indent=4)

    def print_hardware_info(self) -> None:
        """Print hardware capabilities to inform the user of system support."""
        print("\nHardware Capabilities:")
        hw_caps = self.settings['hardware_config']
        print(f"OpenCL Available: {hw_caps.get('OpenCL', False)}")
        print(f"AVX2 Available: {hw_caps.get('AVX2', False)}")
        print(f"AOCL Available: {hw_caps.get('AOCL', False)}")
        print(f"x64 Architecture: {hw_caps.get('x64', False)}")
        time.sleep(1)  # Pause for readability

    def validate_input_file(self, input_path: str) -> bool:
        """Validate the input video file for existence, format, and integrity."""
        if not os.path.exists(input_path):
            print(f"Error: Input file not found: {input_path}")
            time.sleep(5)
            return False
        ext = os.path.splitext(input_path)[1].lower()
        supported_formats = ['.mp4', '.avi', '.mkv']  # Supported video formats
        if ext not in supported_formats:
            print(f"Error: Unsupported format. Supported: {', '.join(supported_formats)}")
            time.sleep(5)
            return False
        try:
            clip = mp_VideoFileClip(input_path)  # Attempt to load video to check integrity
            clip.close()
            return True
        except Exception as e:
            print(f"Error: Invalid video file - {e}")
            time.sleep(5)
            return False

def print_hardware_info(self) -> None:
    print("\nHardware Capabilities:")
    hw_caps = self.settings['hardware_config']  # Consistent pattern
    print(f"OpenCL Available: {hw_caps.get('OpenCL', False)}")
    print(f"AVX2 Available: {hw_caps.get('AVX2', False)}")
    print(f"AOCL Available: {hw_caps.get('AOCL', False)}")
    print(f"x64 Architecture: {hw_caps.get('x64', False)}")
    print("\nHardware Preferences:")
    hw_prefs = self.settings['processing_config']['hardware_acceleration']
    print(f"Use GPU: {hw_prefs.get('use_gpu', False)}")
    print(f"Use OpenCL: {hw_prefs.get('use_opencl', False)}")
    print(f"Use AVX2: {hw_prefs.get('use_avx2', False)}")
    time.sleep(1)

    def validate_input_file(self, input_path: str) -> bool:
        if not os.path.exists(input_path):
            print(f"Error: Input file not found: {input_path}")
            time.sleep(5)  # Error: 5s
            return False
            
        ext = os.path.splitext(input_path)[1].lower()
        supported_formats = ['.mp4', '.avi', '.mkv']
        
        if ext not in supported_formats:
            print(f"Error: Unsupported format. Supported: {', '.join(supported_formats)}")
            time.sleep(5)  # Error: 5s 
            return False
            
        return True

    def process_file(self, input_path: str, output_path: str, target_duration: float) -> None:
        """Process a single video file."""
        try:
            if not self.validate_input_file(input_path):
                return
            
            print(f"Info: Starting processing of {input_path}")
            time.sleep(1)
            result = self.processor.process_video(
                input_path,
                output_path,
                target_duration,
                progress_callback=self._update_progress
            )
            
            if result:
                print(f"Info: Processing completed successfully. Output: {output_path}")
                time.sleep(1)
            else:
                print("Error: Processing failed")
                time.sleep(5)
        except Exception as e:
            print(f"Error: Processing failed - {e}")
            time.sleep(5)

    def _update_progress(self, stage: str, progress: float, message: str) -> None:
        """Update processing progress."""
        print(f"\r{stage}: {progress:.1f}% - {message}", end="")
        # No sleep here to allow real-time progress updates

# Functions...
def print_usage():
    """Print command-line usage information."""
    print("\nMovie Consolidator Usage:")
    print("  Launch GUI:")
    print("    python launcher.py --gui")
    print("  Process single file:")
    print("    python launcher.py input_path output_path target_duration_minutes")
    print("\nExamples:")
    print("  python launcher.py --gui")
    print("  python launcher.py input/game.mp4 output/processed.mp4 120")
    time.sleep(3)  # Important message: 3s

def main():
    try:
        os.system('cls' if os.name == 'nt' else 'clear')
        print("========================================================================")
        print("    Movie Consolidator")
        print("========================================================================")
        time.sleep(1)
        
        print("\nInitializing Program..") 
        consolidator = MovieCompact()
        print("..Initialization Complete.")
        
        print("Program Starting..\n")
        print("Launching GUI interface...")
        time.sleep(1)
        launch_gradio_interface()

    except Exception as e:
        print(f"Critical error: {str(e)}")
        traceback.print_exc()
        time.sleep(5)
        
    finally:
        cleanup_work_directory()
        print("\nCleanup completed")

if __name__ == "__main__":
    print("Entering `main()`...")
    main()
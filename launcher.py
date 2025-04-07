# launcher.py

# Imports...
print("Initializing Imports..")
import os, sys, json, time, traceback
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
        from scripts.temporary import (
            PROCESSING_CONFIG,
            MEMORY_CONFIG,
            SCENE_CONFIG
        )
        
        self.core = CoreUtilities()
        self.settings = load_settings()
        self.hardware_config = self.settings.get('hardware_config', {})
        self.processing_config = PROCESSING_CONFIG
        self.memory_config = MEMORY_CONFIG
        
        self.required_dirs = ["data", "data/temp", "input", "output"]
        self.required_files = [
            os.path.join("data", "persistent.json"),
            os.path.join("data", "requirements.txt"),
            os.path.join("scripts", "__init__.py")
        ]
        
        self.validate_environment()
        
        # Pass settings to components
        self.processor = VideoProcessor(settings=self.settings)
        self.analyzer = VideoAnalyzer(settings=self.settings)
        self.audio_analyzer = AudioAnalyzer()
        self.scene_manager = SceneManager(scene_config=SCENE_CONFIG)
        self.preview_generator = PreviewGenerator()

    def validate_environment(self) -> None:
        try:
            for dir_name in self.required_dirs:
                os.makedirs(dir_name, exist_ok=True)
                print(f"Info: Verified directory: {dir_name}")
                time.sleep(1)  # Normal message: 1s

            missing_files = [f for f in self.required_files if not os.path.exists(f)]
            if missing_files:
                print(f"Error: Missing required files: {', '.join(missing_files)}")
                time.sleep(5)  # Error: 5s
                print("Please run the installer (option 2) first.")
                time.sleep(3)  # Important message: 3s
                sys.exit(1)

        except Exception as e:
            self.error_handler.handle_error(e, "environment_validation")
            sys.exit(1)

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
        # Remove duplicate initialization
        os.system('cls' if os.name == 'nt' else 'clear')
        print("========================================================================")
        print("    Movie Consolidator")
        print("========================================================================")
        time.sleep(1)
        
        # Single initialization call
        print("Initializing Program..") 
        consolidator = MovieCompact()
        print("..Initialization Complete.")
        
        print("Program Starting..")
        if len(sys.argv) > 1:
            if sys.argv[1] == "--gui":
                print("\nLaunching GUI interface...")
                time.sleep(1)
                launch_gradio_interface()
            # CLI mode
            elif len(sys.argv) == 4:
                input_path = sys.argv[1]
                output_path = sys.argv[2]
                try:
                    target_duration = float(sys.argv[3]) * 60  # Convert to seconds
                except ValueError:
                    print("Error: Target duration must be a number")
                    time.sleep(5)  # Error: 5s
                    print_usage()
                    return
                
                consolidator.process_file(input_path, output_path, target_duration)
            else:
                print("Error: Invalid number of arguments")
                time.sleep(5)  # Error: 5s
                print_usage()
        else:
            print("\nLaunching GUI interface...")
            time.sleep(1)
            launch_gradio_interface() 
            
    except KeyboardInterrupt:
        print("\nInfo: Operation cancelled by user")
        time.sleep(1)  # Normal message: 1s
        
    except Exception as e:
        print(f"Error: Program execution failed - {e}")
        traceback.print_exc()  # Added for full stack trace
        time.sleep(5)  # Error: 5s
        
    finally:
        cleanup_work_directory()
        print("\nCleanup completed")
        time.sleep(1)  # Normal message: 1s

if __name__ == "__main__":
    print("Entering `main()`...")
    main()
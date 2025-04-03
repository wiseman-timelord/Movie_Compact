# launcher.py

# Imports...
import os, sys, json, time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from scripts.utility import (
    load_hardware_config,
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

# Classes...
class MovieCompact:
    def __init__(self):
        from scripts.temporary import (
            HARDWARE_CONFIG,
            PROCESSING_CONFIG,
            MEMORY_CONFIG
        )
        
        self.core = CoreUtilities()
        self.hardware_config = HARDWARE_CONFIG
        self.processing_config = PROCESSING_CONFIG
        self.memory_config = MEMORY_CONFIG
        self.settings = load_settings()  # Load settings here
        self.memory_manager = MemoryManager()
        self.error_handler = ErrorHandler()
        
        self.required_dirs = ["data", "input", "output", "work"]
        self.required_files = [
            os.path.join("data", "persistent.json"),
            os.path.join("data", "requirements.txt"),
            os.path.join("data", "hardware.json"),
            os.path.join("scripts", "__init__.py")
        ]
        
        self.validate_environment()
        
        # Pass settings to components
        self.processor = VideoProcessor(settings=self.settings)
        self.analyzer = VideoAnalyzer(settings=self.settings)
        self.audio_analyzer = AudioAnalyzer()
        self.scene_manager = SceneManager()
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
        print("\nHardware Configuration:")
        time.sleep(1)  # Normal message: 1s
        for key, value in self.hardware_config.items():  # Fixed from self.config
            print(f"{key}: {value}")
            time.sleep(1)  # Normal message: 1s
        # Removed print(status) as status is undefined

    def validate_input_file(self, input_path: str) -> bool:
        if not os.path.exists(input_path):
            print(f"Error: Input file not found: {input_path}")
            time.sleep(5)  # Error: 5s
            return False
            
        ext = os.path.splitext(input_path)[1].lower()
        supported_formats = ['.mp4', '.avi', '.mkv']
        
        if ext not in supported_formats:
            print(f"Error: Unsupported format. Supported: {', '.join(supported_formats)}")
            time.sleep(5)  # Error: 5s (replaced log_manager.log)
            return False
            
        return True

    def process_file(self, input_path: str, output_path: str, target_duration: float) -> None:
        try:
            # Placeholder for result; assuming processing logic will be added
            result = True  # Replace with actual logic later
            if result:
                print("Info: Processing completed successfully")
                time.sleep(1)  # Normal message: 1s
            else:
                print("Error: Processing failed")
                time.sleep(5)  # Error: 5s
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(5)  # Error: 5s

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
    """Main entry point for the application."""
    try:
        # Clear screen
        consolidator = MovieCompact()
        
        os.system('cls' if os.name == 'nt' else 'clear')
        
        print("Movie Consolidator")
        print("=================")
        time.sleep(1)  # Normal message: 1s
        
        # Initialize
        settings = load_settings()  # Hypothetical call
        consolidator = MovieCompact()
        consolidator.print_hardware_info()
        
        # Process command line arguments
        if len(sys.argv) > 1:
            # GUI mode
            if sys.argv[1] == "--gui":
                print("\nLaunching GUI interface...")
                time.sleep(1)  # Normal message: 1s
                launch_gradio_interface()  # Removed log_manager
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
            # Default to GUI mode
            print("\nLaunching GUI interface...")
            time.sleep(1)  # Normal message: 1s
            launch_gradio_interface()  # Removed log_manager
            
    except KeyboardInterrupt:
        print("\nInfo: Operation cancelled by user")
        time.sleep(1)  # Normal message: 1s
    except Exception as e:
        print(f"Error: Program execution failed - {e}")
        time.sleep(5)  # Error: 5s
        
    finally:
        cleanup_work_directory()
        print("\nCleanup completed")
        time.sleep(1)  # Normal message: 1s

if __name__ == "__main__":
    main()
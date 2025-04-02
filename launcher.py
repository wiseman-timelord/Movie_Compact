# launcher.py

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
from scripts.manager import download_file, get_file_name_from_url
from scripts.setup import setup_menu, load_config, save_config

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
        self.memory_manager = MemoryManager()
        self.error_handler = ErrorHandler()
        
        # Initialize environment
        self.validate_environment()
        
        # Initialize components
        self.processor = VideoProcessor(self.log_manager)
        self.analyzer = VideoAnalyzer(self.log_manager)
        self.audio_analyzer = AudioAnalyzer()
        self.scene_manager = SceneManager()
        self.preview_generator = PreviewGenerator()

    def validate_environment(self) -> None:
        try:
            for dir_name in required_dirs:
                os.makedirs(dir_name, exist_ok=True)
                print(f"Info: Verified directory: {dir_name}")  # Replaced log_manager.log
                time.sleep(1)

            if missing_files:
                print(f"Error: Missing required files: {', '.join(missing_files)}")  # Replaced log_manager.log
                time.sleep(5)
                print("Please run the installer (option 2) first.")
                sys.exit(1)

        except Exception as e:
            self.error_handler.handle_error(e, "environment_validation")
            sys.exit(1)

    def print_hardware_info(self) -> None:
        info_lines = ["\nHardware Configuration:"]
        for key, value in self.config.hardware_config.items():
            info_line = f"{key}: {value}"
            info_lines.append(info_line)
            print(info_line)  # Replaced log_manager.log
            time.sleep(1)
        
        print(status)  # Replaced log_manager.log
        time.sleep(1)

    def validate_input_file(self, input_path: str) -> bool:
        if not os.path.exists(input_path):
            print(f"Error: Input file not found: {input_path}")  # Replaced log_manager.log
            time.sleep(5)
            return False
            
        ext = os.path.splitext(input_path)[1].lower()
        supported_formats = ['.mp4', '.avi', '.mkv']
        
        if ext not in supported_formats:
            self.log_manager.log(
                f"Unsupported format. Supported: {', '.join(supported_formats)}",
                "ERROR",
                "VALIDATION"
            )
            return False
            
        return True

    def process_file(self, input_path: str, output_path: str, target_duration: float) -> None:
        try:
            if result:
                print("Info: Processing completed successfully")  # Replaced log_manager.log
                time.sleep(1)
            else:
                print("Error: Processing failed")  # Replaced log_manager.log
                time.sleep(5)
        except Exception as e:
            print(f"Error: {e}")  # Replaced generic exception
            time.sleep(5)

    def _update_progress(self, stage: str, progress: float, message: str) -> None:
        """Update and log processing progress."""
        # Remove log_manager.log call
        print(f"\r{stage}: {progress:.1f}% - {message}", end="")

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

def main():
    """Main entry point for the application."""
    try:
        # Clear screen
        os.system('cls' if os.name == 'nt' else 'clear')
        
        print("Movie Consolidator")
        print("=================")
        
        # Initialize
        consolidator = MovieCompact()
        consolidator.print_hardware_info()
        
        # Process command line arguments
        if len(sys.argv) > 1:
            # GUI mode
            if sys.argv[1] == "--gui":
                print("\nLaunching GUI interface...")
                launch_gradio_interface(consolidator.log_manager)
            # CLI mode
            elif len(sys.argv) == 4:
                input_path = sys.argv[1]
                output_path = sys.argv[2]
                try:
                    target_duration = float(sys.argv[3]) * 60  # Convert to seconds
                except ValueError:
                    print("Error: Target duration must be a number")
                    print_usage()
                    return
                
                consolidator.process_file(input_path, output_path, target_duration)
            else:
                print("Error: Invalid number of arguments")
                print_usage()
        else:
            # Default to GUI mode
            print("\nLaunching GUI interface...")
            launch_gradio_interface(consolidator.log_manager)
            
    except KeyboardInterrupt:
        print("\nInfo: Operation cancelled by user")  # Replaced log_event
        time.sleep(1)
    except Exception as e:
        print(f"Error: Program execution failed - {e}")  # Replaced log_event
        time.sleep(5)
        
    finally:
        cleanup_work_directory()
        print("\nCleanup completed")

if __name__ == "__main__":
    main()
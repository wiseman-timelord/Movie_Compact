# launcher.py

import os
import sys
import json
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from utility import (
    load_hardware_config,
    load_settings,
    log_event,
    cleanup_work_directory,
    AudioAnalyzer,
    SceneManager,
    PreviewGenerator,
    LogManager,
    MemoryManager,
    ErrorHandler,
    CoreUtilities
)
from scripts.process import VideoProcessor
from scripts.interface import launch_gradio_interface
from scripts.analyze import VideoAnalyzer

@dataclass
class SystemConfig:
    """System configuration settings."""
    hardware_config: Dict[str, bool]
    settings: Dict[str, Any]
    work_dir: str = "work"
    output_dir: str = "output"
    input_dir: str = "input"
    data_dir: str = "data"

class MovieCompact:
    """Main application class coordinating all components."""
    
    def __init__(self):
        self.core = CoreUtilities()
        self.config = SystemConfig(
            hardware_config=load_hardware_config(),
            settings=load_settings()
        )
        self.log_manager = LogManager(os.path.join("data", "events.txt"))
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
        """Validate and setup program environment."""
        try:
            # Create required directories
            required_dirs = ['data', 'input', 'output', 'work']
            for dir_name in required_dirs:
                os.makedirs(dir_name, exist_ok=True)
                self.log_manager.log(
                    f"Verified directory: {dir_name}",
                    "INFO",
                    "STARTUP"
                )

            # Check required files
            required_files = [
                os.path.join('data', 'temporary.py'),
                os.path.join('data', 'persistent.json'),
                os.path.join('data', 'hardware.txt'),
                os.path.join('data', 'events.txt')
            ]
            
            missing_files = []
            for file_path in required_files:
                if not os.path.exists(file_path):
                    missing_files.append(file_path)
                    # Create events.txt if missing
                    if file_path.endswith('events.txt'):
                        with open(file_path, 'w') as f:
                            f.write("Log initialized\n")
                        continue
            
            if missing_files:
                self.log_manager.log(
                    f"Missing required files: {', '.join(missing_files)}",
                    "ERROR",
                    "STARTUP"
                )
                print("Please run the installer (option 2) first.")
                sys.exit(1)

        except Exception as e:
            self.error_handler.handle_error(e, "environment_validation")
            sys.exit(1)

    def print_hardware_info(self) -> None:
        """Display detected hardware capabilities."""
        info_lines = ["\nHardware Configuration:"]
        for key, value in self.config.hardware_config.items():
            info_line = f"{key}: {value}"
            info_lines.append(info_line)
            self.log_manager.log(info_line, "INFO", "HARDWARE")
        
        if self.config.hardware_config.get("OpenCL"):
            status = "Using OpenCL for GPU acceleration"
        elif self.config.hardware_config.get("Avx2"):
            status = "Using AVX2 for CPU acceleration"
        else:
            status = "Using standard CPU processing"
            
        info_lines.append(status)
        self.log_manager.log(status, "INFO", "HARDWARE")
        print("\n".join(info_lines))

    def validate_input_file(self, input_path: str) -> bool:
        """Validate input file exists and is supported."""
        if not os.path.exists(input_path):
            self.log_manager.log(f"Input file not found: {input_path}",
                               "ERROR", "VALIDATION")
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

    def process_file(self, input_path: str, output_path: str,
                    target_duration: float) -> None:
        """Process a single video file."""
        try:
            if not self.validate_input_file(input_path):
                return

            # Create output directory if needed
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Process the video
            result = self.processor.process_video(
                input_path,
                output_path,
                target_duration,
                progress_callback=self._update_progress
            )

            if result:
                print(f"\nProcessing completed successfully!")
                print(f"Output saved to: {output_path}")
                self.log_manager.log("Processing completed successfully",
                                   "INFO", "PROCESSING")
            else:
                print("\nProcessing failed. Check the logs for details.")
                self.log_manager.log("Processing failed", "ERROR", "PROCESSING")

        except Exception as e:
            self.error_handler.handle_error(e, "file_processing")
            print(f"\nError: {e}")
        finally:
            cleanup_work_directory()

    def _update_progress(self, stage: str, progress: float, message: str) -> None:
        """Update and log processing progress."""
        self.log_manager.log(
            f"Progress [{stage}]: {progress:.1f}% - {message}",
            "INFO",
            "PROGRESS"
        )
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
        print("\nOperation cancelled by user")
        log_event("Operation cancelled by user", "INFO", "CONTROL")
        cleanup_work_directory()
        
    except Exception as e:
        log_event(f"Program execution failed: {e}", "ERROR", "CONTROL")
        print(f"\nError: {e}")
        cleanup_work_directory()
        sys.exit(1)
        
    finally:
        cleanup_work_directory()
        print("\nCleanup completed")

if __name__ == "__main__":
    main()
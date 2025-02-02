# launcher.py

import os
import sys
import json
from typing import Dict, Any, Optional, List
from utility import (
    load_hardware_config,
    load_settings,
    log_event,
    cleanup_work_directory,
    AudioAnalyzer,
    SceneManager,
    PreviewGenerator,
    LogManager  # Added for enhanced logging
)
from scripts.process import VideoProcessor
from scripts.interface import launch_gradio_interface, GradioInterface
from scripts.analyze import VideoAnalyzer

class MovieCompact:
    """Main application class coordinating all components."""
    
    def __init__(self):
        self.settings = load_settings()
        self.hardware_config = load_hardware_config()
        self.log_manager = LogManager(os.path.join("data", "events.txt"))
        self.validate_environment()
        
        # Initialize components with enhanced logging
        self.processor = VideoProcessor(self.log_manager)
        self.analyzer = VideoAnalyzer(self.log_manager)
        self.audio_analyzer = AudioAnalyzer()
        self.scene_manager = SceneManager()
        self.preview_generator = PreviewGenerator()
        
        # Initialize processing state
        self.current_process = None
        self.processing_stage = None
        
        log_event("MovieCompact initialized with enhanced logging", "INFO", "STARTUP")

    def validate_environment(self) -> None:
        """Validate the program environment and required files."""
        try:
            # Check required directories with enhanced logging
            required_dirs = ['data', 'input', 'output', 'work']
            for dir_name in required_dirs:
                if not os.path.exists(dir_name):
                    os.makedirs(dir_name)
                    self.log_manager.log(
                        f"Created missing directory: {dir_name}", 
                        "INFO", 
                        "STARTUP"
                    )

            # Check required files with enhanced status tracking
            required_files = [
                os.path.join('data', 'temporary.py'),
                os.path.join('data', 'persistent.json'),
                os.path.join('data', 'hardware.txt'),
                os.path.join('data', 'events.txt')  # Added for logging
            ]
            
            missing_files = []
            for file_path in required_files:
                if not os.path.exists(file_path):
                    missing_files.append(file_path)
                    # Create events.txt if it's missing
                    if file_path.endswith('events.txt'):
                        with open(file_path, 'w') as f:
                            f.write("Log initialized\n")
                        continue
            
            if missing_files:
                error_msg = f"Missing required files: {', '.join(missing_files)}"
                self.log_manager.log(error_msg, "ERROR", "STARTUP")
                print(f"Error: {error_msg}")
                print("Please run the installer (option 2) first.")
                sys.exit(1)
                    
        except Exception as e:
            error_msg = f"Environment validation failed: {e}"
            self.log_manager.log(error_msg, "ERROR", "STARTUP")
            print(f"Error: {error_msg}")
            sys.exit(1)

    def print_hardware_info(self) -> None:
        """Display detected hardware capabilities with enhanced logging."""
        info_lines = ["\nHardware Configuration:"]
        for key, value in self.hardware_config.items():
            info_line = f"{key}: {value}"
            info_lines.append(info_line)
            self.log_manager.log(info_line, "INFO", "HARDWARE")
        
        if self.hardware_config["OpenCL"]:
            status = "Using OpenCL for GPU acceleration"
        elif self.hardware_config["Avx2"]:
            status = "Using AVX2 for CPU acceleration"
        else:
            status = "Using standard CPU processing"
            
        info_lines.append(status)
        self.log_manager.log(status, "INFO", "HARDWARE")
        
        print("\n".join(info_lines))

    def validate_input_file(self, input_path: str) -> bool:
        """Validate input file exists and is supported with enhanced logging."""
        if not os.path.exists(input_path):
            error_msg = f"Input file not found: {input_path}"
            self.log_manager.log(error_msg, "ERROR", "VALIDATION")
            print(f"Error: {error_msg}")
            return False
            
        ext = os.path.splitext(input_path)[1].lower()
        supported_formats = self.settings.get('video', {}).get(
            'supported_formats', ['.mp4', '.avi', '.mkv']
        )
        
        if ext not in supported_formats:
            error_msg = f"Unsupported file format. Supported: {', '.join(supported_formats)}"
            self.log_manager.log(error_msg, "ERROR", "VALIDATION")
            print(f"Error: {error_msg}")
            return False
            
        self.log_manager.log(f"Validated input file: {input_path}", "INFO", "VALIDATION")
        return True

    def process_file(self, input_path: str, output_path: str, target_duration: float) -> None:
        """Process a single video file with enhanced logging and status tracking."""
        try:
            if not self.validate_input_file(input_path):
                return
                
            # Create output directory if needed
            output_dir = os.path.dirname(output_path)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                self.log_manager.log(
                    f"Created output directory: {output_dir}", 
                    "INFO", 
                    "PROCESSING"
                )
                
            # Update processing state
            self.current_process = {
                'input_file': input_path,
                'output_file': output_path,
                'target_duration': target_duration,
                'start_time': time.time()
            }
            
            self.log_manager.log(
                f"Processing file: {input_path} -> {output_path}", 
                "INFO", 
                "PROCESSING"
            )
            print(f"\nProcessing: {os.path.basename(input_path)}")
            print(f"Target duration: {target_duration/60:.1f} minutes")
            
            # Process the video with progress updates
            result = self.processor.process_video(
                input_path, 
                output_path, 
                target_duration,
                progress_callback=self.update_progress
            )
            
            if result:
                success_msg = (
                    f"\nProcessing completed successfully!\n"
                    f"Output saved to: {output_path}"
                )
                print(success_msg)
                self.log_manager.log(
                    "Processing completed successfully", 
                    "INFO", 
                    "PROCESSING"
                )
            else:
                error_msg = "\nProcessing failed. Check the logs for details."
                print(error_msg)
                self.log_manager.log(
                    "Processing failed", 
                    "ERROR", 
                    "PROCESSING"
                )
                
        except Exception as e:
            error_msg = f"File processing failed: {e}"
            self.log_manager.log(error_msg, "ERROR", "PROCESSING")
            print(f"\nError: {error_msg}")
            
        finally:
            # Reset processing state
            self.current_process = None
            self.processing_stage = None
            cleanup_work_directory()

    def update_progress(self, stage: str, progress: float, message: str) -> None:
        """Update and log processing progress."""
        self.processing_stage = stage
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
    """Main entry point with enhanced error handling and logging."""
    try:
        # Clear screen
        os.system('cls' if os.name == 'nt' else 'clear')
        
        print("Movie Consolidator")
        print("=================")
        
        # Initialize the consolidator
        consolidator = MovieCompact()
        consolidator.print_hardware_info()
        
        # Load settings
        settings = load_settings()
        
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
                    target_duration = float(sys.argv[3]) * 60  # Convert minutes to seconds
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
        error_msg = f"Program execution failed: {e}"
        log_event(error_msg, "ERROR", "CONTROL")
        print(f"\nError: {error_msg}")
        cleanup_work_directory()
        sys.exit(1)
        
    finally:
        cleanup_work_directory()
        print("\nCleanup completed")

if __name__ == "__main__":
    main()
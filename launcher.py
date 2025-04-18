# Script: `.\launcher.py`

# Imports
print("Initializing Imports..")
import os, sys, time, traceback
from moviepy.editor import VideoFileClip as mp_VideoFileClip
from typing import Dict, Any, Optional, List
from scripts.utility import (
    cleanup_work_directory,
    AudioAnalyzer,
    PreviewGenerator,
    MemoryManager,
    ErrorHandler,
    CoreUtilities
)
from scripts.process import VideoProcessor
from scripts.interface import launch_gradio_interface
from scripts.analyze import VideoAnalyzer
from scripts.temporary import ConfigManager, get_full_path
from scripts.hardware import HardwareManager  # NEW IMPORT
print("..Imports Initialized.")

# Classes
class MovieCompact:
    def __init__(self):
        self.hardware_ctx = HardwareManager.create_context()
        self.core = CoreUtilities(self.hardware_ctx)
        self._init_configurations()
        self._validate_environment()
        self._init_components()

    def _init_configurations(self):
        """REPLACES manual config loading"""
        self.supported_formats = ConfigManager.get('processing', 'supported_formats')
        self.required_dirs = [
            get_full_path('data'),
            get_full_path('work'),
            get_full_path('input'),
            get_full_path('output')
        ]
        self.required_files = [
            os.path.join(get_full_path('data'), "persistent.json"),
            os.path.join(get_full_path('data'), "requirements.txt"),
            os.path.join("scripts", "__init__.py")
        ]

    def _init_components(self):
        """REPLACES manual component initialization"""
        self.analyzer = VideoAnalyzer(hardware_ctx=self.hardware_ctx)
        self.processor = VideoProcessor(
            hardware_ctx=self.hardware_ctx,
            analyzer=self.analyzer
        )
        self.audio_analyzer = AudioAnalyzer()
        self.scene_manager = SceneManager()
        self.preview_generator = PreviewGenerator()

    def _validate_environment(self) -> None:
        """UPDATED with dynamic path handling"""
        for path in self.required_dirs:
            os.makedirs(path, exist_ok=True)
            if not os.path.exists(path):
                print(f"Created directory: {path}")
                time.sleep(0.5)

        for file_path in self.required_files:
            if not os.path.exists(file_path):
                self._create_file_resource(file_path)
                print(f"Created file: {file_path}")
                time.sleep(0.5)

    def _create_file_resource(self, path: str):
        """REPLACES _create_default_settings"""
        if "persistent.json" in path:
            with open(path, 'w') as f:
                json.dump({"first_run": True}, f, indent=4)
        else:
            open(file_path, 'w').close()

    def print_hardware_info(self) -> None:
        """UPDATED with HardwareManager integration"""
        caps = HardwareManager.detect_capabilities()
        print("\nHardware Capabilities:")
        print(f"OpenCL Available: {caps['OpenCL']}")
        print(f"AVX2 Available: {caps['AVX2']}")
        print(f"Architecture: {'x64' if caps['x64'] else 'x86'}")
        print("\nActive Acceleration:")
        print(f"Using OpenCL: {self.hardware_ctx['use_opencl']}")
        print(f"Using AVX2: {self.hardware_ctx['use_avx2']}")
        time.sleep(1)

    def validate_input_file(self, input_path: str) -> bool:
        """UPDATED with dynamic format checking"""
        if not os.path.exists(input_path):
            print(f"Error: File not found: {input_path}")
            time.sleep(2)
            return False
            
        ext = os.path.splitext(input_path)[1].lower()
        if ext not in self.supported_formats:
            print(f"Error: Unsupported format. Supported: {', '.join(self.supported_formats)}")
            time.sleep(3)
            return False
            
        try:
            with mp_VideoFileClip(input_path) as clip:
                return clip.duration > 0
        except Exception as e:
            print(f"Invalid video file: {str(e)}")
            time.sleep(3)
            return False

    def process_file(self, input_path: str, output_path: str, target_duration: float) -> None:
        """Streamlined processing flow"""
        if not self.validate_input_file(input_path):
            return

        try:
            print(f"Starting processing: {os.path.basename(input_path)}")
            success = self.processor.process_video(
                input_path,
                output_path,
                target_duration,
                progress_callback=self._update_progress
            )
            
            if success:
                print(f"\nOutput saved to: {output_path}")
            else:
                print("Processing failed")
        except Exception as e:
            print(f"Critical error: {str(e)}")
            traceback.print_exc()

    def _update_progress(self, stage: str, progress: float, message: str) -> None:
        """Real-time progress updates"""
        print(f"\r{stage}: {progress:.1f}% - {message}", end="")

# Functions
def print_usage():
    """Updated usage instructions"""
    print("\nUsage:")
    print("  GUI Mode: python launcher.py --gui")
    print("  CLI Mode: python launcher.py [input] [output] [minutes]")
    print("\nExamples:")
    print("  python launcher.py --gui")
    print("  python launcher.py input/game.mp4 output/processed.mp4 120")
    time.sleep(2)

def main():
    try:
        os.system('cls' if os.name == 'nt' else 'clear')
        print("Movie Consolidator - Action Video Processor")
        print("-------------------------------------------")
        
        consolidator = MovieCompact()
        consolidator.print_hardware_info()
        
        if "--gui" in sys.argv:
            launch_gradio_interface()
        else:
            if len(sys.argv) < 4:
                print_usage()
                return
                
            input_path, output_path, duration = sys.argv[1], sys.argv[2], float(sys.argv[3])
            consolidator.process_file(input_path, output_path, duration*60)

    except Exception as e:
        print(f"Fatal error: {str(e)}")
        traceback.print_exc()
    finally:
        cleanup_work_directory()
        print("\nCleanup completed")

if __name__ == "__main__":
    main()
# Movie_Compact
Status: Alpha (Initial, Testing & Bugfixing, close to beta)

### Plan...
The program is supposed to consolidate very large video clips into action-packed consolidated movie clips. The implementation includes:
1a) Real-time logging system with live updates displayed in the main "Consolidate" page text box, using `.\data\events.txt`. The display auto-refreshes to show progress during processing.
1b) Advanced file analysis that evaluates input files, determining path, length, filesize, and calculates required reduction ratios.
2) Intelligent preview system that converts videos to 360p in the `work` folder, enabling fast frame analysis while maintaining full quality for final output. This dual-processing approach prevents memory overload.
3) Static frame detection system that identifies and removes paused screens, menus, and message popups.
4a) Smart fast-forwarding system for menu navigation scenes while preserving motion context.
4b) Dynamic speed adjustment for low-activity sections, with configurable thresholds (default >10 seconds).
5a) Real-time length tracking system that monitors video duration throughout processing.
5b) Advanced scene detection that identifies content-similar sections.
5c) Audio-based action detection system that identifies high-intensity sequences through noise analysis.
5d) Variable speed processing system that dynamically adjusts playback speeds:
    - Normal speed at scene starts/ends
    - Gradual speed increase to scene midpoints
    - Speed reduction for scene transitions
    - Real-time playback for action sequences
6) Final video compilation with flexible duration targeting (±30 minutes acceptable variance for long videos).
7) Clean output ready for movie maker integration with titles/intro/credits.

## Description
A sophisticated video processing application that creates condensed, action-focused versions of gaming videos using advanced motion, texture, and audio analysis. Features a user-friendly Gradio interface with real-time progress tracking and comprehensive controls.

## Features
- **Enhanced Gradio Interface**: Real-time progress monitoring and control system
- **Multi-Modal Analysis**: Combined motion, texture, and audio analysis for scene detection
- **Dynamic Speed Control**: Smart speed adjustment system with smooth transitions
- **Configurable Parameters**: Comprehensive settings for all processing aspects
- **Universal Format Support**: Handles MP4, AVI, MKV video formats
- **Hardware Acceleration**: Optimized processing using AVX2 and OpenCL
- **Memory Management**: Efficient handling of large video files
- **Real-time Monitoring**: Live progress tracking and status updates

## Preview
- [Screenshots to be added showing interface and processing stages]

## Requirements
- **Python 3.8+**
- **Core Libraries**: 
  - OpenCV (video processing)
  - moviepy (video editing)
  - Gradio (interface)
  - librosa (audio analysis)
- **OS**: Windows (optimized for gaming systems)
- **Hardware**: 
  - x64 architecture
  - AVX2 support
  - OpenCL compatible GPU
  - Minimum 8GB RAM recommended

## Usage
1. **Installation**:
   - Clone the repository
   - Run batch as Administrator
   - Select option 2 (installer.py)
   - Verify installation completion

2. **Running the Program**:
   - Launch batch as Administrator
   - Select option 1 (launcher.py)
   - Open browser to Gradio interface
   - Configure processing settings
   - Monitor consolidation progress
   - Review output file

## Notation
Movie Consolidator complements Movie Maker by handling large (>6 hour) video files effectively, enabling seamless post-processing in Movie Maker.

## File Structure
```
.\
├── README.md             # Project documentation
├── launcher.py           # Main program entry point
├── scripts\
│   ├── interface.py     # Gradio UI implementation
│   ├── analyze.py       # Video/audio analysis
│   ├── process.py       # Video processing engine
│   ├── utility.py       # Support functions/classes

```

## Files Created
```
├── data\               # Configuration and logging
│   ├── events.txt     # Progress and debug logs
│   ├── temporary.py   # Runtime settings
│   ├── persistent.json # User preferences
│   ├── hardware.txt   # System capabilities
├── input\             # Source video storage
├── output\            # Processed video output
├── work\              # Temporary processing files
```

## Credits
- OpenCV: Core video processing
- moviepy: Video manipulation
- Gradio: Interface framework
- librosa: Audio analysis

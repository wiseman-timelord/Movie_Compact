# Movie_Compact
Status: Alpha (code implemented)

## Description
A sophisticated video processing application that creates condensed, action-focused versions of gaming videos using advanced motion, texture, and audio analysis. Features a user-friendly Gradio interface with real-time progress tracking and comprehensive controls.

## Features
- **Enhanced Gradio Interface**: Real-time progress monitoring and control system
- **Multi-Modal Analysis**: Combined motion, texture, and audio analysis for scene detection
- **Dynamic Speed Control**: Smart speed adjustment system with smooth transitions
- **Configurable Parameters**: Comprehensive settings for all processing aspects
- **Universal Format Support**: Handles MP4, AVI, MKV video formats
- **Hardware Acceleration**: Optimized processing using AVX2 and OpenCL.
- **Memory Management**: Efficient handling of large video files
- **Real-time Monitoring**: Live progress tracking and status updates

## Preview
- [Screenshots to be added showing interface and processing stages]

## Requirements
- **Python 3.8+**
- **OS**: Windows v?
- **Hardware**: 
  - x64 architecture
  - AVX2 support
  - OpenCL compatible GPU
  - Minimum 8GB RAM recommended

## Usage
1. Run the batch and select the install process, ensure to check that things install correctly.
2. Affter installation completes successfully, then run the launcher, the interface will then load...
3. TBA.

## Notation
- Movie Consolidator complements Movie Maker by handling large (>6 hour) video files effectively, enabling seamless post-processing in Movie Maker.
- Link for, information about and downloading of, the AVX/2/512 enhancement [AOCL](https://www.amd.com/en/developer/aocl.html) on AMD.Com.

## Development
- At stage of bugfixing Gradio initialization. Need to simplify the Json, for some reason its creating 2 instead of 1. check new document in Notepad++ for current prompt.
- Move `.\work` to `.\data\temp`, inline with my other recent program designs.
- When Initialization bugfixing is complete, then optimize scripts, use more intelligent code to achieve the same result with less overall characters.
- Details of the outline...
```
  1. a) Real-time logging system with live updates displayed in the main "Consolidate" page text box, using `.\data\events.txt`. The display auto-refreshes to show progress during processing.
  1. b) Advanced file analysis that evaluates input files, determining path, length, filesize, and calculates required reduction ratios.
  2. Intelligent preview system that converts videos to 360p in the `work` folder, enabling fast frame analysis while maintaining full quality for final output. This dual-processing approach prevents memory overload.
  3. Static frame detection system that identifies and removes paused screens, menus, and message popups.
  4. a) Smart fast-forwarding system for menu navigation scenes while preserving motion context.
  4. b) Dynamic speed adjustment for low-activity sections, with configurable thresholds (default >10 seconds).
  5. a) Real-time length tracking system that monitors video duration throughout processing.
  5. b) Advanced scene detection that identifies content-similar sections.
  5. c) Audio-based action detection system that identifies high-intensity sequences through noise analysis.
  5. d) Variable speed processing system that dynamically adjusts playback speeds:
      - Normal speed at scene starts/ends
      - Gradual speed increase to scene midpoints
      - Speed reduction for scene transitions
      - Real-time playback for action sequences
  6. ) Final video compilation with flexible duration targeting (±30 minutes acceptable variance for long videos).
  7. ) Clean output ready for example, clips to be loaded in movie maker for then the addition of titles/credits/intro/outro.
```

## Package Structure
```
.\
├── README.md             # Project documentation
├── launcher.py           # Main program entry point
├── requisites.py         # standalone installer script.
├── Movie_Compact.bat     # The Batch launcher for, `launcher.py` and `requisites.py`.
├── scripts\
│   ├── interface.py     # Gradio UI implementation
│   ├── analyze.py       # Video/audio analysis
│   ├── process.py       # Video processing engine
│   ├── utility.py       # Support functions/classes
│   ├── exceptions.py    # Exception Reports
│   ├── temporary.py    # globals and runtime config
```

## Files Created
```
├── scripts\
│   ├── __init__.py     # Blank init file.
├── data\               # Configuration and logging
│   ├── persistent.json # persistent config
├── input\              # Source video storage
├── output\             # Processed video output
├── work\               # Temporary processing files
```

## Credits
- OpenCV: Core video processing
- moviepy: Video manipulation
- Gradio: Interface framework
- librosa: Audio analysis

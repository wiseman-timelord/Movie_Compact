# Movie_Compact
Status: Alpha (code implemented, error on startup, see development)

## Description
A sophisticated video processing application that creates condensed, action-focused feature movies from long-play gaming videos, using advanced motion, texture, and audio analysis. Features a user-friendly Gradio interface with real-time progress tracking and comprehensive controls.

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
- Python => 3.8+ - Libraries are compatible with these versions.
- Windows v? - Assumed v7-v11
- Hardware - Programmed towards, x64, AVX2, OpenCL.
- System Memory - Probably => 16GB RAM.

## Usage
1. Run the batch and select the install process, ensure to check that things install correctly.
2. Affter installation completes successfully, then run the launcher, the interface will then load...
3. TBA.

## Notation
- Movie Consolidator complements Movie Maker by handling large (>6 hour) video files effectively, enabling seamless post-processing in Movie Maker.
- Link for, information about and downloading of, the AVX/2/512 enhancement [AOCL](https://www.amd.com/en/developer/aocl.html) on AMD.Com.

## Development
1. At stage of bugfixing first working version. Repeating issues with error `TypeError: argument of type 'bool' is not iterable` upon startup. The plan is...
- 2 waves of optimizations were done, there is now possibly some missing code, but significantly reduced.
- Assessment of Code, looking for obvious critical issues, then updating. Done once.
- Use Yaml instead, I think it may be a Json issue its blind to?
- Upon failure of above, try when better AI exists/access than, DeepseekR1 and Grok3.

### Outline
- Details of the outline...
```
  1. a) The display auto-refreshes to show progress during processing, with the ability to break from the look via .net.
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
├── data\temp\          # Temporary files folder
├── input\              # Source video storage
├── output\             # Processed video output
```

## Credits
- OpenCV: Core video processing
- moviepy: Video manipulation
- Gradio: Interface framework
- librosa: Audio analysis

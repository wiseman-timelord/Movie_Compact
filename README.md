# Movie Consolidator
Status: Alpha (early version)

## Description
A program that generates condensed, action-packed summaries of gaming videos using motion and texture analysis. It provides a user-friendly Gradio interface to adjust parameters and view results.

## Features
- **Gradio Interface**: A web-based interface for interactive input and output.
- **Motion and Texture Analysis**: Identifies high-action segments in gaming videos.
- **Adjustable Thresholds**: Users can set motion and texture thresholds for analysis.
- **Video Summarization**: Generates condensed video summaries focusing on high-action segments.
- **Supports Multiple Video Formats**: Compatible with various video file types.
- **Efficient Processing**: Optimized for, Avx2 and OpenCL, for processing.

## Preview
- placeholder for screenshots of the interface.

## Requirements
- **Python 3.8 or higher**
- **OpenCV library** for video processing
- **moviepy** for video editing
- **Gradio** for the web interface
- **OS**: Windows, for Windows Gaming. 

## Usage
1. **Installation**:
   - Clone the repository from GitHub.
   - Install required libraries using pip.
2. **Running the Program**:
   - Execute the main script to launch the Gradio interface.
   - Upload a gaming video and adjust the motion and texture thresholds.
   - Generate and download the condensed video summary.

## Notation
- Mention any limitations or issues, such as the need for CUDA for better performance if we decide to add it later.
- The program is named `Movie Consolidator`, because its intended to compliment `Movie Maker`, as `Movie Maker` has issues with >6 hour clips, however, my program intents to go beyond simple consolidation, to take advantage of what is possible.

## File Structure
```
.\
├── README.md             # Project documentation
├── main_script.py        # Main program script
├── scripts\
│   ├── interface.py     # Gradio Interface
│   ├── generate.py      # Video processing and summarization
│   ├── utility.py       # Helper functions for analysis

```

## Files Created
```
├── data\    # for data files
├── output\              # Directory for generated summaries
├── input\  # The user can input files into here for processing optionally.
├── data\temporary.py    # Stores user settings and thresholds.
├── data\persistent.json    # Stores user settings persistently.
├── data\requirements.txt    # for correct install of python libraries.
```

## Development
- **Future plans**:
  - Add support for real-time video summarization.
  - Integrate additional features like audio analysis for better summary accuracy.

## Credits
- **OpenCV**: For video processing functions.
- **moviepy**: For video editing capabilities.
- **Gradio**: For creating the web interface.

## Disclaimer
- This project is currently in development. Features and functionalities are subject to change as development progresses.

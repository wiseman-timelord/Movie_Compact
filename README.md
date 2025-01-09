# Movie Consolidator
Status: Alpha (early version)

### Plan...
```
the current objective for the program is processing to consolidate large clips into shorter more action packed clips...
- cutting frames where the screen is actually paused, for example on a menu or message that popped up.
- fast forwarding on sections of clips where very little changes for over a definable amount of time, for example 15 seconds.
- fast forwarding on menu access of any kind.
- ensuring to include complete sections in real-time for action events, ie gunfire/violence detected.
...if we can do those things, it would be sweet. as a primary processing, and then after that, then...
1) output the new individual videos, as the above processing would be done for each video individually.
2) after updated videos are outputted, then we need to know the new total length of all the videos selected.
3) with the new length known, we would then  be having to detect sections within the videos where there are large sections where the contents is largely the same but there is some motion going on, and these sections would start at normal speed, then be incrementally speeded up to its mid-point, then incrementally slowed to normal speed, for the ending of the relevant "scene", before next section of footage, so as for time to be variable, enabling the remaining amount of footage, to be appropriately fitted-in optimally to the specified time period for the final video. this would be done individually for each video, and the videos will then again be individually outputted, now having optimal lengths, to fit together for within 15 mins for the specified time, either under or over, whatever fits best, thus having 30mins of give for the intended final video length, depending upon how it comes out.
4) with all the videos outputted, the program would then produce the final video, where it has compiled ALL of the videos into one large video to the specified length.
5) the purpose of the program is then complete, and the user may insert the consolidated section of video into movie maker, between some titles and an intro and credits, etc.
```



## Description
A program that generates condensed, action-packed summaries of gaming videos using motion and texture analysis. It provides a user-friendly Gradio interface to adjust parameters and view results, however, my program intents to go beyond simple consolidation of long videos, to taking advantage of what is possible to create great videos.

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
- The program is named `Movie Consolidator`, because its intended to compliment `Movie Maker`, as `Movie Maker` has issues with >6 hour clips.

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

## Credits
- **OpenCV**: For video processing functions.
- **moviepy**: For video editing capabilities.
- **Gradio**: For creating the web interface.

## Disclaimer
- This project is currently in development. Features and functionalities are subject to change as development progresses.

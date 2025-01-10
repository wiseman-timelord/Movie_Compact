# Movie Consolidator
Status: Alpha (early version)

### Plan...
```
the current objective for the program is processing to consolidate, one very large or multiple large clips, into a single action packed consolidated movie clip...
1a) cutting frames where the screen is actually paused, for example on a menu or message that popped up.
1b) fast forwarding on menu access of any kind.
...if we can do those things, it would be sweet. as a primary processing, and the videos will then again be individually outputted,and then after that, then secondary processing...
2a) fast forwarding on sections of clips where extremely little changes for over a definable amount of time, for example >10 seconds.
2b) output the new individual videos, as the above processing would be done for each video individually.
...if we can do those things, as secondary processing, and the videos will then again be individually outputted, and then after that, then tertiary processing...
3a) after updated videos are outputted, then we need to know the new total length currently of all the videos selected, 
3b) with the new length known, we would then  be having to detect sections within the videos where the contents is largely the same, and these sections would be "scenes".
3c) The "scenes" are where we are going to squash down the time, start at normal speed at the start of a given scene, then be incrementally speeded up to its mid-point where it is at most highest speed, then incrementally slowed to normal speed, for the ending of the relevant "scene", before next scene of footage, so as for time to be variable, enabling large scenes to be dynamically compacted, to a degree, that are predicted to consolidated the final combined result, to the specified time period for the final video, however the calculation is best done, but, as are possible to do so, we need to also be ensuring to include complete sections in real-time for action scenes, ie gunfire/violence/explosions detected within scene.
3d) the videos will then again be individually outputted, as the final versions of the videos.
...if we can do those things, as secondary processing, and the videos will then again be individually outputted, and then after that, then fourth phase...
4a) the final versions of the individual movie files, will then be merged in the correct alphabetic order, and the final video will be produced, that are hopefully something around the intended specified length for the video. Ideally having 30mins of give, more or less, around the indended final length would be acceptable for a 6 hour video consolidated from for example 12 hours, but however it ends up fitting best in relevance to the speeding up and down, we will have to fully theorise that part.
4b) with the final video, where it has compiled ALL of the videos into one large video to the specified length, the purpose of the program is then complete,
5) the user may insert the consolidated section of video into movie maker, between some titles and an intro and credits, etc, to produce their polished gaming video.
```
### Notes...
```
First, there's an installer script that sets up the environment. It checks for hardware capabilities like whether the system is 64-bit, supports AVX2 or OpenCL, and records this in a hardware.txt file. It also installs necessary Python packages and sets up default settings in a persistent.json file. That makes sense; setting up the environment is crucial for the program to function correctly.

Next, the utility.py script handles the core processing functions. It includes motion detection using OpenCL for GPU acceleration, which should speed things up if the GPU is capable. There's also a fallback to CPU processing if OpenCL isn't available. Additionally, there's a plan to implement audio analysis using librosa to detect audio events like gunfire, which will help in identifying action scenes.

Then, the generate.py script is supposed to process the videos. It removes paused frames, fast-forwards through uneventful parts, and adjusts the speed of scenes to fit within a specified duration. However, it seems like some parts of this script are still under development, especially the merging of videos and dynamic speed adjustments.

The interface.py script provides a user-friendly interface using Gradio. It allows users to input folder paths containing videos, set target lengths for summaries, choose output qualities, and displays hardware information. This should make it easier for users to interact with the program without needing to dive into the command line.

The launcher.py script is the main entry point. It loads hardware configuration, selects the best processing methods based on hardware, and starts the summarization process. It also handles merging individual video summaries into a final output.

Lastly, the MovieConsolidate.bat script manages the environment setup, checks for admin privileges, sets up a virtual environment, and provides a menu for installing requirements or running the summarization process. This is handy for users who prefer a batch file for managing the program.

Potential challenges include ensuring compatibility across different hardware setups, optimizing performance for various systems, and handling edge cases in video processing like inconsistent frame rates or different codecs. Future enhancements might involve integrating AI for smarter scene selection, which could improve the summarization process.

Overall, the program aims to streamline creating concise, action-packed video summaries, making it easier for users to produce high-quality content with minimal effort.
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

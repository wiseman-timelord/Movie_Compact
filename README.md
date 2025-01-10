# Movie Consolidator
Status: Alpha (early version)

### Plan...
```
the current objective for the program is processing to consolidate, one very large clip, into a single action packed consolidated movie clip, that can then have intro/outro/title/credits added, to make a feature video...
1a) we need some kind of text output logging on the main display, we should log actions and output to a `.\data\events.txt`, that is then displayed in a text box in the main "Consolidate" page, I want this to be a box that people will be watching for updates during the processes, so it will be required to update itself.
1b) the specified input file will be read, and a filename/path known and the total length/filesize of the clip will be assessed, and then the amount to reduce the clip will be calculated.
2) the videos should then be converted into 360p videos outputted by default in the `work` folder, these will later be used for faster assessment of frames/data, but we will be producing relating editing upon the full quality versions, this would be done via knowing the relating times in the video in which to apply the edits, and this way the system memory will not be over-loaded by the full quality video; and then with the preview versions of teh videos created.
3) cutting frames where the screen is actually paused, for example on a menu or message that popped up.
4a) fast forwarding on menu access of any kind, where there is still some motion.
4b) fast forwarding on sections of clips where extremely little changes for over a definable amount of time, for example >10 seconds.
5a) then we need to know the new total length currently of the processed video, because the processes before were somewhat of a cleanup, where the next stage will be intelligently speeding up and slowing down, to a necessary amount, in order to fit the video to the specified length...
5b) with the new length known, we would then  be having to detect sections within the videos where the contents is largely the same, and these sections would be "scenes". 
5b) We also need to find the sections where there there is more noise, these would be action scenesinvolving battles such as gunfire/violence/explosions.
5c) The "scenes" are where we are going to squash down the time, start at normal speed at the start of a given scene, then be incrementally speeded up to its mid-point where it is at most highest speed, then incrementally slowed to normal speed, for the ending of the relevant "scene", before next scene of footage, so as for time to be variable, enabling large scenes to be dynamically compacted, to a degree, that are predicted to consolidated the final combined result, to the specified time period for the final video, however the calculation is best done, but, as are possible to do so, we need to also be ensuring to include complete real-time sections for the action scenes. we will have to fully theorise that part.
6) After all the editing, then the final video will be produced, that are hopefully something around the intended specified length for the video. Ideally having 30mins of give, more or less, around the indended final length would be acceptable for, for example, a 10 hour video consolidated to 2-4 hours, but however it ends up fitting best, it does not need to be precise.
7) with the final video, where it has compiled ALL of the editing into one action packed video to the specified length, the purpose of the program is then complete, the user may insert the consolidated section of video into movie maker, between some titles and an intro and credits, etc, to produce their polished gaming video.
```
### Notes...
```
First, there's an installer script that sets up the environment. It checks for hardware capabilities like whether the system is 64-bit, supports AVX2 or OpenCL, and records this in a hardware.txt file. It also installs necessary Python packages and sets up default settings in a persistent.json file. That makes sense; setting up the environment is crucial for the program to function correctly. It will additionally create

Next, the utility.py script handles the core processing functions. It includes motion detection using OpenCL for GPU acceleration, which should speed things up if the GPU is capable. There's also a fallback to CPU processing if OpenCL isn't available. Additionally, there's a plan to implement audio analysis using librosa to detect audio events like gunfire, which will help in identifying action scenes.

Then, the generate.py script is supposed to process the videos. It removes paused frames, fast-forwards through uneventful parts, and adjusts the speed of scenes to fit within a specified duration. However, it seems like some parts of this script are still under development, especially the merging of videos and dynamic speed adjustments.

The interface.py script provides a user-friendly interface using Gradio. It allows users to input folder paths containing videos, set target lengths for summaries, choose output qualities, and displays hardware information. This should make it easier for users to interact with the program without needing to dive into the command line. the created `.\data\events.txt` will be required to be blanked upon the start of a new project.

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
- **Python Libraries**: installed by installer; OpenCV (video processing), moviepy (video editing), Gradio (web interface).
- **OS**: Windows; Targetting Windows gamers, whom intend to edit the Video in same system/boot. 
- **Hardware**: Compatible with, x64. Avx2, OpenCL. No CUDA, support or plans, currently.

## Usage
1. **Installation**:
   - Clone the repository from GitHub to a suitable location.
   - Run the batch as an Administrator, choosing option 2 on the menu, this will then install, requirements and libraries and setup folders, through `.\installer.py`.
2. **Running the Program**:
   - Run the batch as an Administrator, then choose option 1 on the menu, this will run `.\launcher.py`.
   - Launch the browser and point it to the address of the Gradio interface.
   - Configure the settings in the `Configure` page, then save, and return to `Consolidate` page.
   - assess the information on the `Consolidate` page, then as applicable, click `Start Consolidation`.
   - Watch the output from the processes of consolidation, hopefully it will be verbose.
   - When processing is finished it will also tell you, you should then assess the outputted file.

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
├── output\   # default folder for final consolidated video.
├── input\  # default folder for input files into program.
├── work\  #  default folder for clips in the process of being worked on through stages, after first stage of processing and output, and also place for downsized videos for faster assessment.
├── data\temporary.py    # Stores user settings and thresholds.
├── data\persistent.json    # Stores user settings persistently.
├── data\requirements.txt    # for correct install of python libraries.
├── data\events.txt    # for display of log and debug data.
```

## Credits
- **OpenCV**: For video processing functions.
- **moviepy**: For video editing capabilities.
- **Gradio**: For creating the web interface.

## Disclaimer
- This project is currently in development. Features and functionalities are subject to change as development progresses.

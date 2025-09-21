# PianoSheetia
A Python tool for converting Synthesia-style piano videos into MIDI files using computer vision.

The system processes videos frame-by-frame, detecting which piano keys are pressed based on brightness changes from established baselines, then generates properly timed MIDI files that can be played back or imported into music software.

## Main Components
**SheetConverter** - Main orchestrator class that handles the complete conversion pipeline from video input to MIDI output. Coordinates all other components and manages the conversion workflow.

**KeyboardDetector** - Computer vision engine that locates piano keyboards in video frames using template matching. Calculates precise key positions and samples brightness values at each key location for press detection.

**PianoKeyboard** & **PianoKey** - Data structures representing a complete 88-key piano layout. Stores key properties (note names, colors, positions) and mutable detection data (brightness values, baselines).

**MidiGenerator** - Converts key press events into properly timed MIDI format with dynamic velocity calculation based on brightness changes. Handles note on/off events and maintains video-synced tempo.

**VideoDownloader** - Handles both YouTube URLs and local video files using yt-dlp. Downloads videos to local storage and manages file path resolution.

**KeyboardVisualizer** - Creates debug visualizations showing detected piano boundaries and key positions for validation and troubleshooting.

## Key Features
* Multi-scale template matching for robust piano detection across different video sizes
* Multi-point sampling strategy that tests multiple frames to find the best keyboard detection
* Brightness-based key press detection using statistical baselines (IQR method)
* Dynamic velocity calculation based on brightness change intensity
* Video-synced MIDI timing with automatic tempo detection
* Validation system including middle C pattern verification
* Progress tracking with visual progress bars during processing
* Automatic filename generation based on input video names

## Supported Video Types
* YouTube URLs (automatically downloaded using yt-dlp)
* Local video files (.mp4, .avi, .mov, and other OpenCV-supported formats)

## Installation
```bash
# Requires Python 3.8+
pip3 install -r requirements.txt
```
## How to use
```bash
# Basic usage with YouTube URL
python main.py https://youtu.be/f43rKPkB2qw

# Basic usage with local video file
python main.py synthesia_video.mp4

# With custom activation threshold (higher = less sensitive)
python main.py --act-threshold 40 https://youtu.be/f43rKPkB2qw

# With custom piano template
python main.py --template template/your_template.png video.mp4
```
Output will be automatically saved as `output/video_name.mid` based on the input video filename.

## Converting MIDI to Sheet Music
To convert the generated .mid file to sheet music, use music notation software:
* MuseScore (free) - Open the .mid file directly
* Finale - Import MIDI file
* Sibelius - Import MIDI file

## Command Line Options
```bash
positional arguments:
  video                       YouTube URL or local video file (.mp4)

optional arguments:
  -h, --help                  show this help message and exit
  --act-threshold THRESHOLD   Activation threshold for key press detection
                              (default: 30)
  --template TEMPLATE         Path to piano template file for detection
                              (default: template/piano-88-keys.png)
```
<!--
## Unit Testing
```bash
python -m unittest -v tests/test_piano_keyboard.py
```
-->

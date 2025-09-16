"""
PianoSheetia - Synthesia Piano Videos to Sheet Music

A Python package for converting synthesia piano videos into sheet music.
"""

# Import main classes and functions
from .piano_keyboard import PianoKeyboard, PianoKey
from .video_downloader import VideoDownloader
#from .keyboard_detector import KeyboardDetector

# Package metadata
__version__ = "0.1.0"
__author__ = "Yi-Chen Zhang"
__email__ = "chris7462@gmail.com"
__description__ = "Convert synthesia piano videos to sheet music"

# Define what gets imported with "from VidToSheet import *"
__all__ = [
    "PianoKey",
    "PianoKeyboard",
    "VideoDownloader",
#    "KeyboardDetector",
]

# Optional: Add convenience imports for common use cases
# from .processor import VidToSheetProcessor  # When you create this later

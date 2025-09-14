"""
VidToSheet - Video to Sheet Music Conversion Tool

A Python package for converting piano performance videos into sheet music
by detecting piano keys and analyzing their positions over time.
"""

# Import main classes and functions
#from .keyboard_detector import PianoKeyDetector
from .piano_keyboard import PianoKeyboard, PianoKey
#from .video_processor import VideoProcessor

# Package metadata
__version__ = "0.1.0"
__author__ = "Yi-Chen Zhang"
__email__ = "chris7462@gmail.com"
__description__ = "Convert piano performance videos to sheet music"

# Define what gets imported with "from VidToSheet import *"
__all__ = [
    #"PianoKeyDetector",
    "PianoKey",
    "PianoKeyboard",
    #"VideoProcessor",
]

# Optional: Add convenience imports for common use cases
# from .processor import VidToSheetProcessor  # When you create this later
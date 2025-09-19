"""
sheet_converter.py

Provides the SheetConverter class for converting piano videos to MIDI files
by analyzing key presses through computer vision.
"""

import cv2
from typing import Optional, List
from tqdm import tqdm

from .video_downloader import VideoDownloader
from .keyboard_detector import KeyboardDetector
from .piano_keyboard import PianoKeyboard
from .midi_generator import MidiGenerator

from .keyboard_visualizer import create_detection_visualization


class SheetConverter:
    """
    Main orchestrator for converting piano videos to MIDI files.

    Handles the complete conversion process from video input to MIDI output,
    including keyboard detection and frame processing. MIDI generation is
    delegated to the MidiGenerator class.
    """

    def __init__(self, activation_threshold: int = 20,
                 template_path: str = "data/template/piano-88-keys-0_5.png",
                 show_progress: bool = True):
        """
        Initialize the piano converter.

        Args:
            activation_threshold: Brightness change threshold for key press detection
            template_path: Path to piano template file for keyboard detection
            show_progress: Whether to show progress bar during conversion
        """
        self.activation_threshold = activation_threshold
        self.template_path = template_path
        self.show_progress = show_progress

        # Initialize components
        self.video_downloader = VideoDownloader()
        self.keyboard = PianoKeyboard()
        self.detector = KeyboardDetector(template_path)
        self.midi_generator = None  # Will be created once we know the FPS

    def convert(self, video_path: str, output_path: str = "output/out.mid") -> bool:
        """
        Convert a piano video to MIDI file.

        Args:
            video_path: YouTube URL or local video file path
            output_path: Output MIDI file path

        Returns:
            True if conversion successful, False otherwise
        """
        try:
            print(f"Starting conversion: {video_path} -> {output_path}")

            # Setup video capture
            video_capture = self._setup_video(video_path)
            if video_capture is None:
                return False

            # Get video info
            fps = video_capture.get(cv2.CAP_PROP_FPS)
            total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps
            print(f'Processing entire video: {duration:.1f} seconds at {fps:.1f} fps ({total_frames} frames)...')

            # Initialize MIDI generator with FPS
            self.midi_generator = MidiGenerator(fps)

            # Process first frame for keyboard detection
            success, first_frame = video_capture.read()
            if not success:
                print("Could not read first frame from video")
                video_capture.release()
                return False

            if not self._detect_keyboard_layout(first_frame):
                video_capture.release()
                return False

            # Reset video to beginning and process all frames
            video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

            # Process all frames with progress bar
            success = True
            frame_count = 0

            # Create progress bar
            if self.show_progress:
                pbar = tqdm(total=total_frames, desc="Processing frames", unit="frame")

            try:
                while success:
                    success, image = video_capture.read()
                    if not success:
                        break

                    # Process frame and generate MIDI events
                    current_pressed = self._process_frame(image)
                    if current_pressed is not None:
                        self.midi_generator.process_frame(current_pressed)

                    frame_count += 1

                    # Update progress bar
                    if self.show_progress:
                        pbar.update(1)

            finally:
                # Always close progress bar
                if self.show_progress:
                    pbar.close()

            # Cleanup and save
            video_capture.release()

            if self.midi_generator.save(output_path):
                print(f"Conversion complete! Saved as {output_path}")
                return True
            else:
                print("Failed to save MIDI file")
                return False

        except Exception as e:
            print(f"Conversion failed: {e}")
            return False

    def _setup_video(self, video_path: str) -> Optional[cv2.VideoCapture]:
        """Setup video capture from file or URL."""
        # Get video file (handles both URLs and local files)
        input_video = self.video_downloader.get_video_file(video_path)
        if input_video is None:
            print("Failed to get video file")
            return None

        video_capture = cv2.VideoCapture(input_video)
        if not video_capture.isOpened():
            print(f"Could not open video: {input_video}")
            return None

        return video_capture

    def _detect_keyboard_layout(self, first_frame) -> bool:
        """Detect keyboard layout on the first frame."""
        print("Detecting keyboard layout...")

        if not self.detector.detect(first_frame, self.keyboard):
            print("Failed to detect keyboard. Please check:")
            print("1. Template file exists and is valid")
            print("2. Video contains a visible piano")
            print("3. Piano is clearly visible in the first frame")
            return False

        print("Keyboard detection successful!")

        # Create visualization of detected keys using the visualization function
        create_detection_visualization(
            image=first_frame,
            keyboard=self.keyboard,
            piano_boundary=self.detector.piano_boundary,
            output_path="output/keyboard_detection.jpg"
        )

        return True

    def _process_frame(self, image) -> Optional[List[int]]:
        """
        Process a single frame and return current key states.

        Args:
            image: Current frame image

        Returns:
            List of key states (0/1) or None if processing failed
        """
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray_image = image

            # Sample brightness at all key positions using the detector
            self.detector.sample_brightness(gray_image, self.keyboard)

            # Detect and return pressed keys
            return self._get_pressed_keys()

        except Exception as e:
            print(f"Error processing frame: {e}")
            return None

    def _get_pressed_keys(self) -> List[int]:
        """Determine which keys are currently pressed based on brightness changes from baseline."""
        pressed = []
        for key in self.keyboard:
            if key.color == 'W':
                # White key - compare to white baseline
                if abs(key.brightness - self.keyboard.white_baseline) > self.activation_threshold:
                    pressed.append(1)
                else:
                    pressed.append(0)
            else:  # Black key
                # Black key - compare to black baseline
                if abs(key.brightness - self.keyboard.black_baseline) > self.activation_threshold:
                    pressed.append(1)
                else:
                    pressed.append(0)
        return pressed


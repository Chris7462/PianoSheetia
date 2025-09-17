"""
sheet_converter.py

Provides the PianoConverter class for converting piano videos to MIDI files
by analyzing key presses through computer vision.
"""

import cv2
from mido import Message, MidiFile, MidiTrack
from typing import Optional, List, Tuple
from tqdm import tqdm

from .video_downloader import VideoDownloader
from .keyboard_detector import KeyboardDetector
from .piano_keyboard import PianoKeyboard
from .keyboard_visualizer import create_detection_visualization


class SheetConverter:
    """
    Main orchestrator for converting piano videos to MIDI files.

    Handles the complete conversion process from video input to MIDI output,
    including keyboard detection, frame processing, and MIDI generation.
    """

    # MIDI constants
    MIDI_VELOCITY_ON = 64
    MIDI_VELOCITY_OFF = 127
    MIDI_TICKS_PER_BEAT = 480

    def __init__(self, activation_threshold: int = 30,
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

        # State for frame processing
        self.last_pressed = []
        self.last_mod = 0

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

            # Setup MIDI file
            midi_file, track = self._setup_midi()

            # Setup video capture
            video_capture = self._setup_video(video_path)
            if video_capture is None:
                return False

            # Get video info
            fps = video_capture.get(cv2.CAP_PROP_FPS)
            total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps
            print(f'Processing entire video: {duration:.1f} seconds at {fps:.1f} fps ({total_frames} frames)...')

            # Process first frame for keyboard detection
            success, first_frame = video_capture.read()
            if not success:
                print("Could not read first frame from video")
                video_capture.release()
                return False

            if not self._detect_keyboard_layout(first_frame):
                video_capture.release()
                return False

            # Initialize frame processing state
            self.last_pressed = [0] * len(self.keyboard)
            self.last_mod = 0

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
                    midi_events = self._process_frame(image, frame_count, fps)

                    # Add MIDI events to track
                    for event in midi_events:
                        track.append(event)

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
            midi_file.save(output_path)
            print(f"Conversion complete! Saved as {output_path}")
            return True

        except Exception as e:
            print(f"Conversion failed: {e}")
            return False

    def _setup_midi(self) -> Tuple[MidiFile, MidiTrack]:
        """Create and setup MIDI file and track."""
        midi_file = MidiFile(ticks_per_beat=self.MIDI_TICKS_PER_BEAT)
        track = MidiTrack()
        midi_file.tracks.append(track)
        return midi_file, track

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

        # Note: Verification is now handled internally by the detector.detect() method
        print("Keyboard detection successful!")

        # Store default brightness values for comparison
        for key in self.keyboard:
            key.default_brightness = key.brightness

        # Create visualization of detected keys using the visualization function
        create_detection_visualization(
            image=first_frame,
            keyboard=self.keyboard,
            piano_boundary=self.detector.piano_boundary,
            output_path="output/keyboard_detection.jpg"
        )

        return True

    def _process_frame(self, image, frame_count: int, fps: float) -> List[Message]:
        """
        Process a single frame and return MIDI events for key state changes.

        Args:
            image: Current frame image
            frame_count: Current frame number
            fps: Video frames per second

        Returns:
            List of MIDI Message objects for key state changes
        """
        # Sample current brightness at all key positions
        self._sample_brightness(image)

        # Detect pressed keys
        current_pressed = self._get_pressed_keys()

        # Generate MIDI events for key state changes
        midi_events = []

        for i, (current_state, last_state) in enumerate(zip(current_pressed, self.last_pressed)):
            if current_state != last_state:
                midi_note = i + 21  # A0 = 21, so key index + 21

                # Calculate timing
                if self.last_mod == 0 and frame_count > fps:
                    self.last_mod = frame_count - fps

                time_delta = int((frame_count - self.last_mod) * (self.MIDI_TICKS_PER_BEAT / fps))

                if current_state == 1:
                    # Note on
                    midi_events.append(Message('note_on', note=midi_note,
                                             velocity=self.MIDI_VELOCITY_ON, time=time_delta))
                else:
                    # Note off
                    midi_events.append(Message('note_off', note=midi_note,
                                             velocity=self.MIDI_VELOCITY_OFF, time=time_delta))

                self.last_mod = frame_count

        # Update state for next frame
        self.last_pressed = current_pressed

        return midi_events

    def _sample_brightness(self, image):
        """Sample current brightness values for all keys."""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = image

        # Sample brightness at each key position
        for key in self.keyboard:
            if key.x is not None and key.y is not None:
                # Ensure coordinates are within image bounds
                y = min(max(0, key.y), gray_image.shape[0] - 1)
                x = min(max(0, key.x), gray_image.shape[1] - 1)
                key.brightness = float(gray_image[y, x])

    def _get_pressed_keys(self) -> List[int]:
        """Determine which keys are currently pressed based on brightness changes."""
        pressed = []
        for key in self.keyboard:
            if key.brightness is None or key.default_brightness is None:
                pressed.append(0)
            elif abs(key.brightness - key.default_brightness) > self.activation_threshold:
                pressed.append(1)
            else:
                pressed.append(0)
        return pressed

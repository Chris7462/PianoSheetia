"""
sheet_converter.py

Provides the PianoConverter class for converting piano videos to MIDI files
by analyzing key presses through computer vision.
"""

import cv2
from mido import Message, MidiFile, MidiTrack
from typing import Optional, List, Tuple

from PianoSheetia import VideoDownloader
from PianoSheetia import KeyboardDetector
from PianoSheetia import PianoKeyboard


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
                 progress_callback=None):
        """
        Initialize the piano converter.

        Args:
            activation_threshold: Brightness change threshold for key press detection
            template_path: Path to piano template file for keyboard detection
            progress_callback: Optional callback function for progress updates (current, total)
        """
        self.activation_threshold = activation_threshold
        self.template_path = template_path
        self.progress_callback = progress_callback or self._default_progress_callback

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

            # Process all frames
            frame_count = 0
            success = True

            # Reset video to beginning and process all frames
            video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

            while success:
                success, image = video_capture.read()
                if not success:
                    break

                # Process frame and generate MIDI events
                midi_events = self._process_frame(image, frame_count, fps)

                # Add MIDI events to track
                for event in midi_events:
                    track.append(event)

                # Update progress
                self.progress_callback(frame_count, total_frames)

                frame_count += 1

            # Cleanup and save
            video_capture.release()
            midi_file.save(output_path)
            print(f"\nConversion complete! Saved as {output_path}")
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

        # Verify detection quality
        try:
            self.detector.verify_middle_c(self.keyboard)
            print("Keyboard detection successful!")
        except ValueError as e:
            print(f"Keyboard detection verification failed: {e}")
            print("Detection may be inaccurate. Consider using a different template.")
            return False

        # Store default brightness values for comparison
        for key in self.keyboard:
            key.default_brightness = key.brightness

        # Create visualization of detected keys
        self._create_detection_visualization(first_frame)

        return True

    def _create_detection_visualization(self, image):
        """Create and save keyboard detection visualization."""
        vis_image = image.copy()

        # Draw piano boundary
        if self.detector.piano_boundary:
            x, y, w, h = self.detector.piano_boundary
            cv2.rectangle(vis_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Draw key positions
        for key in self.keyboard:
            if key.x is not None and key.y is not None:
                color = (255, 255, 255) if key.type == 'W' else (0, 0, 0)
                cv2.circle(vis_image, (key.x, key.y), 3, color, -1)
                cv2.circle(vis_image, (key.x, key.y), 5, (0, 255, 0), 1)

        cv2.imwrite("output/keyboard_detection.jpg", vis_image)
        print("Keyboard detection visualization saved as 'output/keyboard_detection.jpg'")

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
                key = self.keyboard[i]
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

    def _default_progress_callback(self, current: int, total: int):
        """Default progress reporting to console."""
        progress_percent = (current / total) * 100
        print(f"Processing frame {current}/{total} ({progress_percent:.1f}%)...", end="\r")

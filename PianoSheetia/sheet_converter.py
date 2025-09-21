"""
sheet_converter.py

Provides the SheetConverter class for converting piano videos to MIDI files
by analyzing key presses through computer vision.
"""

import cv2
import os
from typing import Optional, List, Tuple
from tqdm import tqdm
from contextlib import contextmanager

from .video_downloader import VideoDownloader
from .keyboard_detector import KeyboardDetector
from .piano_keyboard import PianoKeyboard
from .midi_generator import MidiGenerator
from .keyboard_visualizer import create_detection_visualization


@contextmanager
def progress_bar(total: int, desc: str):
    """Context manager for progress bar handling."""
    pbar = tqdm(total=total, desc=desc, unit="frame")
    try:
        yield pbar
    finally:
        pbar.close()


class SheetConverter:
    """
    Main orchestrator for converting piano videos to MIDI files.

    Handles the complete conversion process from video input to MIDI output,
    including keyboard detection and frame processing. MIDI generation is
    delegated to the MidiGenerator class.
    """

    def __init__(self, activation_threshold: int, template_path: str):
        """
        Initialize the piano converter.

        Args:
            activation_threshold: Brightness change threshold for key press detection
            template_path: Path to piano template file for keyboard detection
        """
        self.activation_threshold = activation_threshold
        self.template_path = template_path

        # Initialize components
        self.video_downloader = VideoDownloader()
        self.keyboard = PianoKeyboard()
        self.detector = KeyboardDetector(template_path)
        self.midi_generator = None  # Will be created once we know the FPS

    def convert(self, video_path: str) -> bool:
        """
        Convert a piano video to MIDI file.

        Args:
            video_path: YouTube URL or local video file path

        Returns:
            True if conversion successful, False otherwise
        """
        print(f"Starting conversion: {video_path}")

        try:
            # Step 1: Setup video capture and get video info
            video_info = self._setup_conversion(video_path)
            if video_info is None:
                return False

            video_capture, fps, total_frames, output_path = video_info

            # Step 2: Detect keyboard layout
            detection_successful = self._perform_keyboard_detection(video_capture, fps, total_frames)
            if not detection_successful:
                video_capture.release()
                return False

            # Step 3: Process all frames to generate MIDI
            self._process_all_frames(video_capture, total_frames)

            # Step 4: Cleanup and save
            video_capture.release()
            self.midi_generator.save(output_path)
            print(f"Conversion complete! Saved as {output_path}")
            return True

        except Exception as e:
            print(f"Conversion failed: {e}")
            return False

    def _setup_conversion(self, video_path: str) -> Optional[Tuple[cv2.VideoCapture, float, int, str]]:
        """
        Setup video capture, extract video information, and generate output filename.

        Args:
            video_path: YouTube URL or local video file path

        Returns:
            Tuple of (video_capture, fps, total_frames, output_path) if successful, None if failed
        """
        # Get video file (handles both URLs and local files)
        input_video = self.video_downloader.get_video_file(video_path)
        if input_video is None:
            print("Failed to get video file")
            return None

        # Generate output filename based on input video filename
        base_name = os.path.splitext(os.path.basename(input_video))[0]
        output_path = f"output/{base_name}.mid"

        # Ensure output directory exists
        os.makedirs("output", exist_ok=True)

        video_capture = cv2.VideoCapture(input_video)
        if not video_capture.isOpened():
            print(f"Could not open video: {input_video}")
            return None

        # Extract video information
        fps = video_capture.get(cv2.CAP_PROP_FPS)
        total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps

        print(f'Processing entire video: {duration:.1f} seconds at {fps:.1f} fps ({total_frames} frames)...')
        print(f'Output will be saved as: {output_path}')

        # Initialize MIDI generator with FPS
        self.midi_generator = MidiGenerator(fps)

        return video_capture, fps, total_frames, output_path

    def _perform_keyboard_detection(self, video_capture: cv2.VideoCapture,
                                    fps: float, total_frames: int) -> bool:
        """
        Detect keyboard layout using multi-point sampling strategy.

        Args:
            video_capture: OpenCV VideoCapture object
            fps: Video frames per second
            total_frames: Total number of frames in video

        Returns:
            True if detection successful, False otherwise
        """
        print("Detecting keyboard layout using multi-point sampling...")

        # Define sampling times (0.0, 0.5, 1.0, ..., 5.0 seconds)
        sampling_times = [i * 0.5 for i in range(11)]  # [0.0, 0.5, 1.0, ..., 5.0]

        best_confidence = 0.0
        best_frame = None
        best_detection_successful = False

        # Early stopping threshold
        early_stop_threshold = 0.85

        for time_seconds in sampling_times:
            # Convert time to frame number
            frame_number = int(time_seconds * fps)

            # Skip if frame number exceeds video length
            if frame_number >= total_frames:
                print(f"Skipping time {time_seconds}s (frame {frame_number}) - beyond video length")
                continue

            # Seek to the target frame
            video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            success, frame = video_capture.read()

            if not success:
                print(f"Failed to read frame at {time_seconds}s (frame {frame_number})")
                continue

            print(f"Testing frame at {time_seconds}s (frame {frame_number})...")

            # Create a temporary keyboard for this detection attempt
            temp_keyboard = PianoKeyboard()

            # Attempt detection on this frame
            result = self.detector.detect(frame, temp_keyboard)
            if result is not None:
                confidence = result.confidence

                print(f"Detection successful at {time_seconds}s with confidence {confidence:.3f}")

                # Update best match if this is better
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_frame = frame.copy()
                    # Copy the successful detection to our main keyboard
                    self.keyboard = temp_keyboard
                    best_detection_successful = True

                # Early stopping if confidence is very high
                if confidence >= early_stop_threshold:
                    print(f"Early stopping: confidence {confidence:.3f} >= {early_stop_threshold}")
                    break

            else:
                print(f"Detection failed at {time_seconds}s")

        if not best_detection_successful:
            print("Keyboard detection failed at all sampled time points.")
            print("The template may not be suitable for this video.")
            return False

        print(f"Best detection found with confidence {best_confidence:.3f}")

        # Create visualization using the best detection frame
        create_detection_visualization(
            image=best_frame,
            keyboard=self.keyboard,
            piano_boundary=self.detector.piano_boundary,
            output_path="output/keyboard_detection.jpg"
        )

        return True

    def _process_all_frames(self, video_capture: cv2.VideoCapture, total_frames: int) -> bool:
        """
        Process all frames in the video to generate MIDI events.

        Args:
            video_capture: OpenCV VideoCapture object
            total_frames: Total number of frames to process

        Returns:
            True if processing successful, False otherwise
        """
        # Reset video to beginning
        video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

        frame_count = 0

        with progress_bar(total_frames, "Processing frames") as pbar:
            while True:
                success, image = video_capture.read()
                if not success:
                    break

                # Process frame and generate MIDI events
                current_pressed = self._process_single_frame(image)
                if current_pressed is None:
                    print(f"Warning: Failed to process frame {frame_count}")
                    # Continue processing rather than failing completely
                    current_pressed = [0] * 88  # Default to no keys pressed

                self.midi_generator.process_frame(current_pressed)
                frame_count += 1

                # Update progress bar
                pbar.update(1)

        print(f"Processed {frame_count} frames successfully")
        return True

    def _process_single_frame(self, image) -> Optional[List[int]]:
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

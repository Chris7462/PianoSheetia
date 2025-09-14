import cv2
import numpy as np
from typing import Tuple, Optional
import os

from .keyboard_detector import PianoKeyboard


class PianoKeyDetector:
    """
    Computer vision detector for locating piano keys in images
    """

    def __init__(self, template_path: str):
        if not template_path:
            raise ValueError("template_path is required and cannot be empty")

        if not os.path.exists(template_path):
            raise FileNotFoundError(f"Template file not found: {template_path}")

        # Template matching setup
        self.template = None
        self.template_gray = None
        self.template_h = 0
        self.template_w = 0

        if template_path and os.path.exists(template_path):
            self.load_template(template_path)

    def load_template(self, template_path: str) -> bool:
        """Load piano template for boundary detection with validation"""
        try:
            self.template = cv2.imread(template_path)
            if self.template is None:
                print(f"Error: Could not load template from {template_path}")
                return False

            self.template_gray = cv2.cvtColor(self.template, cv2.COLOR_BGR2GRAY)
            self.template_h, self.template_w = self.template_gray.shape

            print(f"Template loaded successfully: {self.template_w}x{self.template_h}")
            return True

        except Exception as e:
            print(f"Error loading template: {e}")
            return False

    def detect(self, image: np.ndarray, keyboard: PianoKeyboard) -> None:
        """
        Main detection function that updates keyboard with positions and brightness

        Args:
            image: Input keyboard image
            keyboard: PianoKeyboard object to update

        Raises:
            RuntimeError: If piano boundary detection or key positioning fails
        """

        # Step 1: Detect piano boundary using template matching
        piano_boundary = self.detect_piano_boundary(image)

        if piano_boundary is None:
            raise RuntimeError("Failed to detect piano boundary - cannot proceed with key detection")

        # Convert image to grayscale for brightness sampling
        if len(image.shape) == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = image.copy()

        # Step 2: Calculate positions and update white keys
        piano_x, piano_y, piano_w, piano_h = piano_boundary
        white_key_width = piano_w / keyboard.WHITE_KEYS
        white_y = piano_y + int(piano_h * keyboard.WHITE_KEY_Y_RATIO)

        white_index = 0
        for i, key_type in enumerate(keyboard.key_pattern):
            if key_type == 'W':
                # Calculate white key position
                white_x = piano_x + int((white_index + 0.5) * white_key_width)

                # Sample brightness at this position
                brightness = float(gray_image[white_y, white_x])

                # Update the key
                keyboard[i].x = white_x
                keyboard[i].y = white_y
                keyboard[i].brightness = brightness

                white_index += 1

        # Step 3: Calculate positions and update black keys
        black_y = piano_y + int(piano_h * keyboard.BLACK_KEY_Y_RATIO)

        for i, key_type in enumerate(keyboard.key_pattern):
            if key_type == 'B':
                # Black key is between adjacent white keys
                left_white_key = keyboard[i-1]
                right_white_key = keyboard[i+1]

                if left_white_key.x is not None and right_white_key.x is not None:
                    # TODO: Change it later. The black key is not at the center of two white keys
                    black_x = (left_white_key.x + right_white_key.x) // 2

                    # Sample brightness at this position
                    brightness = float(gray_image[black_y, black_x])

                    # Update the key
                    keyboard[i].x = black_x
                    keyboard[i].y = black_y
                    keyboard[i].brightness = brightness
                else:
                    raise RuntimeError(f"Key position calculation failed for black key at index {i}")

        # Validate that all keys have been positioned
        unpositioned_keys = [i for i, key in enumerate(keyboard) if key.x is None or key.y is None or key.brightness is None]
        if unpositioned_keys:
            raise RuntimeError(f"Failed to position {len(unpositioned_keys)} keys: {unpositioned_keys}")

        print(f"Successfully detected and positioned {len(keyboard)} piano keys")

    def detect_piano_boundary(self, image: np.ndarray, confidence_threshold: float = 0.7) -> Optional[Tuple[int, int, int, int]]:
        """
        Detect piano boundary using robust template matching

        Args:
            image: Input image
            confidence_threshold: Minimum confidence for template match

        Returns:
            Tuple (x, y, width, height) of piano boundary, or None if not found
        """
        if self.template_gray is None:
            print("Error: No template loaded for boundary detection")
            return None

        # Convert image to grayscale
        if len(image.shape) == 3:
            gray_frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray_frame = image.copy()

        # Multi-scale template matching - scales from 1.0 since template is smaller than input pianos
        scales = [round(1.0 + i * 0.1, 1) for i in range(21)]   # generate sequence from 1.0 to 3.0
        best_match = None
        best_confidence = 0
        best_scale = 1.0
        best_template_size = (self.template_w, self.template_h)

        for scale in scales:
            # Calculate scaled template size
            new_w = int(self.template_w * scale)
            new_h = int(self.template_h * scale)

            # Skip if template is larger than image
            if new_h > gray_frame.shape[0] or new_w > gray_frame.shape[1]:
                continue

            try:
                # Resize template
                scaled_template = cv2.resize(self.template_gray, (new_w, new_h))

                # Perform template matching with normalized correlation
                result = cv2.matchTemplate(gray_frame, scaled_template, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(result)

                # Update best match if confidence is higher
                if max_val > best_confidence:
                    best_confidence = max_val
                    best_match = max_loc
                    best_scale = scale
                    best_template_size = (new_w, new_h)

            except cv2.error as e:
                print(f"Template matching error at scale {scale}: {e}")
                continue

        # Validate the best match
        if best_match is None or best_confidence < confidence_threshold:
            print(f"Piano boundary not detected - confidence {best_confidence:.3f} below threshold {confidence_threshold}")
            return None

        x, y = best_match
        w, h = best_template_size

        print(f"Piano boundary detected: confidence={best_confidence:.3f}, scale={best_scale:.2f}, size={w}x{h}")
        return (x, y, w, h)

    def verify_middle_c(self, keyboard: PianoKeyboard) -> bool:
        """
        Verify that the calculated middle C (index 39) is correct using brightness pattern matching.

        Middle C should be a white key, and we verify the surrounding pattern matches piano structure.
        Uses 128 as threshold: white keys >= 128, black keys < 128.

        Args:
            keyboard: PianoKeyboard object with positioned keys and brightness values

        Returns:
            True if middle C at index 39 appears correct

        Raises:
            ValueError: If middle C verification fails
        """
        THRESHOLD = 128
        middle_c_index = 39  # Middle C4 position in 88-key piano (0-indexed)

        # First check: Middle C (index 39) must be a white key
        middle_c = keyboard[middle_c_index]
        if middle_c.brightness is None or middle_c.brightness < THRESHOLD:
            raise ValueError(f"Middle C verification failed: Index 39 brightness ({middle_c.brightness}) is below white key threshold ({THRESHOLD}). This suggests the key positioning is incorrect.")

        # Second check: Verify the 12-key pattern around Middle C (6 keys left + middle C + 5 keys right)
        # This covers a full octave from F#3 to F4, centered on middle C
        # Pattern: indices 34-45 representing F#3-G3-G#3-A3-A#3-B3-C4-C#4-D4-D#4-E4-F4
        # Expected key types: B-W-B-W-B-W-W-B-W-B-W-W (one complete octave pattern)
        expected_pattern = ['B', 'W', 'B', 'W', 'B', 'W', 'W', 'B', 'W', 'B', 'W', 'W']
        pattern_start = 34  # Start from F#3 (5 keys left of middle C)

        for i, expected_type in enumerate(expected_pattern):
            key_index = pattern_start + i
            key = keyboard[key_index]

            if key.brightness is None:
                raise ValueError(f"Middle C verification failed: Key at index {key_index} ({key.name}) has no brightness value.")

            if expected_type == 'W' and key.brightness < THRESHOLD:
                raise ValueError(f"Middle C verification failed: Expected white key at index {key_index} ({key.name}), but brightness ({key.brightness:.1f}) indicates black key.")
            elif expected_type == 'B' and key.brightness >= THRESHOLD:
                raise ValueError(f"Middle C verification failed: Expected black key at index {key_index} ({key.name}), but brightness ({key.brightness:.1f}) indicates white key.")

        print(f"Middle C verification successful: Index 39 ({middle_c.name}) brightness = {middle_c.brightness:.1f}")
        print(f"12-key octave pattern verification successful:")
        for i in range(12):
            key = keyboard[pattern_start + i]
            key_type = "White" if expected_pattern[i] == 'W' else "Black"
            print(f"  {key.name}: {key.brightness:.1f} ({key_type})")

        return True

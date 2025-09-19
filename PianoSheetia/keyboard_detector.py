import cv2
import numpy as np
from typing import Tuple, Optional
import os

from .piano_keyboard import PianoKeyboard


class KeyboardDetector:
    """
    Computer vision detector for locating piano keys in images
    """
    # Key positioning constants (as ratios of piano height)
    _white_key_y_ratio = 0.75  # White keys in lower portion of piano
    _black_key_y_ratio = 0.35  # Black keys in upper portion of piano
    _jnd = 5 # Just noticeable difference (JND) for perceptual is about 5-10 levels

    def __init__(self, template_path: str):
        if not template_path:
            raise ValueError("template_path is required and cannot be empty")

        if not os.path.exists(template_path):
            raise FileNotFoundError(f"Template file not found: {template_path}")

        # Template matching setup
        self.template = None
        self._template_gray = None
        self._template_h = 0
        self._template_w = 0

        # Piano boundary storage
        self.piano_boundary = None  # Stores (x, y, width, height) of detected piano boundary

        # Load template immediately since path is validated
        self._load_template(template_path)

    def detect(self, image: np.ndarray, keyboard: PianoKeyboard) -> bool:
        """
        Main detection function that updates keyboard with positions and brightness

        Args:
            image: Input keyboard image
            keyboard: PianoKeyboard object to update

        Returns:
            bool: True if detection is successful, False otherwise
        """
        try:
            # Convert to grayscale once for all processing
            if len(image.shape) == 3:
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray_image = image.copy()

            # Step 1: Detect piano boundwhite_brightnessary using template matching
            self.piano_boundary = self._detect_piano_boundary(gray_image)
            if self.piano_boundary is None:
                print("Failed to detect piano boundary - cannot proceed with key detection")
                return False

            # Step 2: Calculate key positions
            self._calculate_key_positions(keyboard)

            # Step 3: Sample brightness values
            self.sample_brightness(gray_image, keyboard)

            # Step 4: Validate layout
            return self._verify_layout(keyboard)

        except Exception as e:
            print(f"Error during key detection: {e}")
            return False

    def sample_brightness(self, gray_image: np.ndarray, keyboard: PianoKeyboard) -> None:
        """
        Sample brightness values at key positions and calculate baseline values

        Args:
            gray_image: Grayscale input image
            keyboard: PianoKeyboard object to update with brightness values and baselines
        """
        half = 2

        for key in keyboard:
            # Sample 5x5 grid and take average for brightness at this position
            region = gray_image[key.y-half:key.y+half+1, key.x-half:key.x+half+1]
            mean_brightness = np.mean(region)

            if key.color == 'W':  # White keys: bias toward brighter values
                brightness = np.ceil(mean_brightness).astype(int).item()
            else:  # Black keys: bias toward darker values
                brightness = np.floor(mean_brightness).astype(int).item()

            # Update the key brightness
            key.brightness = brightness

        # Calculate baseline values using IQR approach
        self._calculate_baselines(keyboard)

    def _load_template(self, template_path: str) -> bool:
        """Load piano template for boundary detection with validation"""
        try:
            self.template = cv2.imread(template_path)
            if self.template is None:
                print(f"Error: Could not load template from {template_path}")
                return False

            self._template_gray = cv2.cvtColor(self.template, cv2.COLOR_BGR2GRAY)
            self._template_h, self._template_w = self._template_gray.shape

            print(f"Template loaded successfully: {self._template_w}x{self._template_h}")
            return True

        except Exception as e:
            print(f"Error loading template: {e}")
            return False

    def _calculate_baselines(self, keyboard: PianoKeyboard) -> None:
        """
        Calculate baseline brightness values for white and black keys using IQR approach

        Args:
            keyboard: PianoKeyboard object to update with baseline values
        """
        # Collect brightness values for white and black keys separately
        white_brightness = [key.brightness for key in keyboard if key.color == 'W']
        black_brightness = [key.brightness for key in keyboard if key.color == 'B']

        # Calculate white key baseline using IQR
        if white_brightness:
            white_brightness.sort()
            n = len(white_brightness)
            q1_idx = n // 4
            q3_idx = (3 * n) // 4

            # Take average of values between Q1 and Q3 (inclusive)
            iqr_values = white_brightness[q1_idx:q3_idx + 1]
            keyboard.white_baseline = round(np.ceil(sum(iqr_values) / len(iqr_values)))

        # Calculate black key baseline using IQR
        if black_brightness:
            black_brightness.sort()
            n = len(black_brightness)
            q1_idx = n // 4
            q3_idx = (3 * n) // 4

            # Take average of values between Q1 and Q3 (inclusive)
            iqr_values = black_brightness[q1_idx:q3_idx + 1]
            keyboard.black_baseline = round(np.floor(sum(iqr_values) / len(iqr_values)))

    def _detect_piano_boundary(self, gray_image: np.ndarray, confidence_threshold: float = 0.7) -> Optional[Tuple[int, int, int, int]]:
        """
        Detect piano boundary using robust template matching

        Args:
            image: Input grayscale image
            confidence_threshold: Minimum confidence for template match

        Returns:
            Tuple (x, y, width, height) of piano boundary, or None if not found
        """
        if self._template_gray is None:
            print("Error: No template loaded for boundary detection")
            return None

        # Multi-scale template matching - scales from 1.0 since template is smaller than input pianos
        scales = [round(1.0 + i * 0.1, 1) for i in range(21)]   # generate sequence from 1.0 to 3.0
        best_match = None
        best_confidence = 0
        best_scale = 1.0
        best_template_size = (self._template_w, self._template_h)

        for scale in scales:
            # Calculate scaled template size
            new_w = int(self._template_w * scale)
            new_h = int(self._template_h * scale)

            # Skip if template is larger than image
            if new_h > gray_image.shape[0] or new_w > gray_image.shape[1]:
                continue

            try:
                # Resize template
                scaled_template = cv2.resize(self._template_gray, (new_w, new_h))

                # Perform template matching with normalized correlation
                result = cv2.matchTemplate(gray_image, scaled_template, cv2.TM_CCOEFF_NORMED)
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

    def _calculate_key_positions(self, keyboard: PianoKeyboard) -> None:
        """
        Calculate positions for all keys based on detected piano boundary

        Args:
            keyboard: PianoKeyboard object to update with x, y coordinates
        """
        # Calculate white key positions first
        piano_x, piano_y, piano_w, piano_h = self.piano_boundary
        white_key_width = piano_w / keyboard.white_key_count
        white_y = piano_y + int(piano_h * self._white_key_y_ratio)

        keyboard_key_colors = keyboard.get_key_colors()

        # Position white keys
        white_index = 0
        for i, key_type in enumerate(keyboard_key_colors):
            if key_type == 'W':
                # Calculate white key position
                white_x = piano_x + int((white_index + 0.5) * white_key_width)

                # Update the key position
                keyboard[i].x = white_x
                keyboard[i].y = white_y

                white_index += 1

        # Position black keys
        black_y = piano_y + int(piano_h * self._black_key_y_ratio)

        for i, key_type in enumerate(keyboard_key_colors):
            if key_type == 'B':
                # Black key is between adjacent white keys
                left_white_key = keyboard[i-1]
                right_white_key = keyboard[i+1]

                # Determine offset based on the black key type
                note_letter = keyboard[i].name[0]  # Get first letter (C, D, F, G, A)

                # Magic numbers. Tuned by human integillent, not AI. Don't touch this section!
                match note_letter:
                    case 'C':
                        black_x = int(left_white_key.x * 0.53 + right_white_key.x * 0.47)
                    case 'D':
                        black_x = int(left_white_key.x * 0.35 + right_white_key.x * 0.65)
                    case 'F':
                        black_x = int(left_white_key.x * 0.62 + right_white_key.x * 0.38)
                    case 'G':
                        black_x = (left_white_key.x + right_white_key.x) // 2
                    case 'A':
                        black_x = int(left_white_key.x * 0.3 + right_white_key.x * 0.7)

                # Update the key position
                keyboard[i].x = black_x
                keyboard[i].y = black_y

    def _verify_layout(self, keyboard: PianoKeyboard) -> bool:
        """
        Verify the detected keyboard layout using multiple validation checks

        Args:
            keyboard: PianoKeyboard object to validate

        Returns:
            bool: True if layout verification passes, False otherwise
        """
        # Check 1: Validate detection completeness
        if not self._validate_detection_completeness(keyboard):
            return False

        # Check 2: Validate baseline calculations
        if not self._validate_baselines(keyboard):
            return False

        # Check 3: Verify middle C and surrounding pattern
        try:
            return self._verify_middle_c(keyboard)
        except ValueError as e:
            print(f"Layout verification failed: {e}")
            return False

    def _validate_detection_completeness(self, keyboard: PianoKeyboard) -> bool:
        """
        Validate that all keys have been properly positioned and have brightness values

        Args:
            keyboard: PianoKeyboard object to validate

        Returns:
            bool: True if all keys are complete, False otherwise
        """
        unpositioned_keys = [i for i, key in enumerate(keyboard)
                             if key.x is None or key.y is None or key.brightness is None]

        if unpositioned_keys:
            print(f"Failed to position {len(unpositioned_keys)} keys: {unpositioned_keys}")
            return False

        return True

    def _validate_baselines(self, keyboard: PianoKeyboard) -> bool:
        """
        Validate that baseline brightness values have been properly calculated

        Args:
            keyboard: PianoKeyboard object to validate

        Returns:
            bool: True if baselines are valid, False otherwise
        """
        # Check if white baseline is calculated
        if keyboard.white_baseline is None:
            print("Layout verification failed: White key baseline not calculated")
            return False

        # Check if black baseline is calculated
        if keyboard.black_baseline is None:
            print("Layout verification failed: Black key baseline not calculated")
            return False

        # Check it baseline values are reasonable (0-255 for 8-bit grayscale)
        if not (0 <= keyboard.white_baseline <= 255):
            print(f"Layout verification failed: White baseline ({keyboard.white_baseline}) out of valid range [0-255]")
            return False

        if not (0 <= keyboard.black_baseline <= 255):
            print(f"Layout verification failed: Black baseline ({keyboard.black_baseline}) out of valid range [0-255]")
            return False

        # Check if white keys are generally brighter than black keys
        if keyboard.white_baseline <= keyboard.black_baseline:
            print(f"Layout verification warning: White baseline ({keyboard.white_baseline}) "
                  f"should typically be brighter than black baseline ({keyboard.black_baseline})")
            return False

        print(f"Baseline validation successful: White={keyboard.white_baseline}, Black={keyboard.black_baseline}")
        return True

    def _verify_middle_c(self, keyboard: PianoKeyboard) -> bool:
        """
        Verify that the calculated middle C (index 39) is correct using brightness pattern matching.

        Middle C should be a white key, and we verify the surrounding pattern matches piano structure.

        Args:
            keyboard: PianoKeyboard object with positioned keys and brightness values

        Returns:
            bool: True if middle C verification passes, False otherwise
        """

        # First check: Middle C (index 39) must be a white key
        middle_c = keyboard.find_key_by_name('C4')

        if abs(middle_c.brightness - keyboard.white_baseline) > self._jnd:
            print(f"Middle C verification failed: Middle C brightness ({middle_c.brightness}) "
                  f"is outside of the white key baseline ({keyboard.white_baseline} ± {self._jnd}). "
                  f"This suggests the key positioning is incorrect.")
            return False

        # Second check: Verify the 12-key pattern around Middle C (6 keys left + middle C + 5 keys right)
        # This covers a full octave from F#3 to F4, centered on middle C
        # Pattern: indices 33-44 representing F#3-G3-G#3-A3-A#3-B3-C4-C#4-D4-D#4-E4-F4
        # Expected key types: B-W-B-W-B-W-W-B-W-B-W-W (one complete octave pattern)
        expected_colors = ['B', 'W', 'B', 'W', 'B', 'W', 'W', 'B', 'W', 'B', 'W', 'W']
        pattern_start = keyboard.find_key_by_name('F#3').index  # Start from F#3 (5 keys left of middle C)

        for i, expected_color in enumerate(expected_colors):
            key_index = pattern_start + i
            key = keyboard[key_index]

            # Sanity check: verify the theoretical key color matches expected pattern
            if key.color != expected_color:
                print(f"Middle C verification failed: Key at index {key_index} ({key.name}) "
                      f"has color '{key.color}' but expected '{expected_color}'. "
                      f"This indicates a problem with the piano layout or pattern definition.")
                return False

            # Check if key matches expected color based on brightness
            if expected_color == 'W':
                if abs(key.brightness - keyboard.white_baseline) > self._jnd:
                    print(f"Middle C verification failed: Expected white key at index {key_index} "
                          f"({key.name}), but brightness ({key.brightness}) is outside of the "
                          f"white baseline ({keyboard.white_baseline} ± {self._jnd})")
                    return False
            else:  # expected_color == 'B'
                if abs(key.brightness - keyboard.black_baseline) > self._jnd:
                    print(f"Middle C verification failed: Expected black key at index {key_index} "
                          f"({key.name}), but brightness ({key.brightness}) is outside of the "
                          f"black baseline ({keyboard.black_baseline} ± {self._jnd})")
                    return False

        print(f"Middle C verification successful: Middle C ({middle_c.name}) brightness = {middle_c.brightness}")
        print(f"12-key octave pattern verification successful")
        return True

import cv2
import numpy as np
from typing import List, Tuple, Optional
import os


class PianoKeyDetector:
    """
    Optimized piano key detector using template matching and mathematical positioning:
    1. Template matching to locate piano boundary accurately
    2. Mathematical calculation of all 88 key positions
    3. Focus on precision for image-based piano key detection
    4. Extract brightness values at key positions
    """

    # Piano structure constants
    TOTAL_KEYS = 88
    WHITE_KEYS = 52
    BLACK_KEYS = 36
    NOTES_PER_OCTAVE = 12
    COMPLETE_OCTAVES = 7
    KEYS_BEFORE_C1 = 3  # A0, A#0, B0
    # Key positioning constants (as ratios of piano height)
    WHITE_KEY_Y_RATIO = 0.75  # White keys in lower portion of piano
    BLACK_KEY_Y_RATIO = 0.35  # Black keys in upper portion of piano

    def __init__(self, template_path: str):
        if not template_path:
            raise ValueError("template_path is required and cannot be empty")

        if not os.path.exists(template_path):
            raise FileNotFoundError(f"Template file not found: {template_path}")

        # Standard 88-key piano structure (A0 to C8)
        # Pattern for one octave: W-B-W-B-W-W-B-W-B-W-B-W
        self.octave_pattern = ['W', 'B', 'W', 'B', 'W', 'W', 'B', 'W', 'B', 'W', 'B', 'W']
        self.note_names = ['A', 'A#', 'B', 'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#']

        # Build the complete 88-key pattern
        self.key_pattern = self.build_88_key_pattern()

        # Indices for white and black keys
        self.white_key_indices = [i for i, k in enumerate(self.key_pattern) if k == 'W']  # 52 white keys
        self.black_key_indices = [i for i, k in enumerate(self.key_pattern) if k == 'B']  # 36 black keys

        # Template matching setup
        self.template = None
        self.template_gray = None
        self.template_h = 0
        self.template_w = 0

        if template_path and os.path.exists(template_path):
            self.load_template(template_path)

        print(f"Piano structure: {len(self.white_key_indices)} white keys, {len(self.black_key_indices)} black keys")

    def build_88_key_pattern(self) -> List[str]:
        """Build the complete 88-key piano pattern starting from A0"""
        # 88-key piano: A0, A#0, B0, then 7 full octaves (C1 to B7), then C8
        pattern = ['W', 'B', 'W']   # A0, A#0, B0

        # Add complete octaves (C1 to B7)
        pattern += self.octave_pattern * self.COMPLETE_OCTAVES

        # Add final C8
        pattern.append('W')

        return pattern

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

    def detect_keys(self, image: np.ndarray) -> Tuple[List[Tuple[int, int]], List[float]]:
        """
        Main detection function that returns both key positions and brightness values

        Args:
            image: Input keyboard image

        Returns:
            Tuple of (key_positions, brightness_values)
            - key_positions: List of (x, y) coordinates for all 88 keys
            - brightness_values: List of brightness values at each key position
        """

        # Step 1: Detect piano boundary using template matching
        piano_boundary = self.detect_piano_boundary(image)

        if piano_boundary is None:
            print("Failed to detect piano boundary - cannot proceed with key detection")
            return [], []

        # Step 2: Calculate all key positions mathematically
        key_positions = self.calculate_all_key_positions(piano_boundary)

        # Step 3: Extract brightness values at key positions
        brightness_values = self.extract_brightness_values(image, key_positions)

        print(f"Successfully detected {len(key_positions)} piano keys with brightness values")
        return key_positions, brightness_values

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
        #scales = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.7, 2.0, 2.5, 3.0]
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

    def calculate_all_key_positions(self, piano_boundary: Tuple[int, int, int, int]) -> List[Tuple[int, int]]:
        """
        Calculate all 88 key positions mathematically using piano boundary

        Args:
            piano_boundary: (x, y, width, height) of piano region

        Returns:
            List of (x, y) coordinates for all 88 keys in order
        """
        piano_x, piano_y, piano_w, piano_h = piano_boundary

        # Calculate white key positions (evenly distributed across piano width)
        white_key_width = piano_w / self.WHITE_KEYS
        white_positions = []

        white_y = piano_y + int(piano_h * self.WHITE_KEY_Y_RATIO)  # White keys in lower portion
        black_y = piano_y + int(piano_h * self.BLACK_KEY_Y_RATIO)  # Black keys in upper portion

        # Generate all white key positions
        for i in range(self.WHITE_KEYS):
            white_x = piano_x + int((i + 0.5) * white_key_width)
            white_positions.append((white_x, white_y))

        # Initialize array for all key positions
        all_key_positions = [None] * self.TOTAL_KEYS

        # Place white keys first using the pattern
        white_index = 0
        for i, key_type in enumerate(self.key_pattern):
            if key_type == 'W':
                all_key_positions[i] = white_positions[white_index]
                white_index += 1

        # Now place black keys between appropriate white keys
        for i, key_type in enumerate(self.key_pattern):
            if key_type == 'B':
                # Black key is always between i-1 (white) and i+1 (white)
                left_white_pos = all_key_positions[i-1]
                right_white_pos = all_key_positions[i+1]

                if left_white_pos and right_white_pos:
                    # TODO: Change it later. The black key is not at the center of two white keys
                    black_x = (left_white_pos[0] + right_white_pos[0]) // 2
                    all_key_positions[i] = (black_x, black_y)

        # Validate that all key positions were calculated
        none_count = all_key_positions.count(None)
        if none_count > 0:
            print(f"Error: {none_count} key positions were not calculated")

        return all_key_positions

    def extract_brightness_values(self, image: np.ndarray, key_positions: List[Tuple[int, int]]) -> List[float]:
        """
        Extract brightness values at key positions

        Args:
            image: Input image (color or grayscale)
            key_positions: List of (x, y) coordinates for keys

        Returns:
            List of brightness values corresponding to each key position
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = image.copy()

        brightness_values = []

        for x, y in key_positions:
            # Extract brightness value at the key position
            brightness = float(gray_image[y, x])
            brightness_values.append(brightness)

        return brightness_values

    def verify_middle_c(self, brightness_values: List[float]) -> bool:
        """
        Verify that the calculated middle C (index 39) is correct using brightness pattern matching.

        Middle C should be a white key, and we verify the surrounding pattern matches piano structure.
        Uses 128 as threshold: white keys >= 128, black keys < 128.

        Args:
            brightness_values: List of 88 brightness values from extract_brightness_values

        Returns:
            True if middle C at index 39 appears correct

        Raises:
            ValueError: If middle C verification fails
        """
        if len(brightness_values) != self.TOTAL_KEYS:
            raise ValueError(f"Expected {self.TOTAL_KEYS} brightness values, got {len(brightness_values)}")

        THRESHOLD = 128
        middle_c_index = 39  # Middle C4 position in 88-key piano (0-indexed)

        # First check: Middle C (index 39) must be a white key
        middle_c_brightness = brightness_values[middle_c_index]
        if middle_c_brightness < THRESHOLD:
            raise ValueError(f"Middle C verification failed: Index 39 brightness ({middle_c_brightness:.1f}) is below white key threshold ({THRESHOLD}). This suggests the key positioning is incorrect.")

        # Second check: Verify the 12-key pattern around Middle C (6 keys left + middle C + 5 keys right)
        # This covers a full octave from F#3 to F4, centered on middle C
        # Pattern: indices 34-45 representing F#3-G3-G#3-A3-A#3-B3-C4-C#4-D4-D#4-E4-F4
        # Expected key types: B-W-B-W-B-W-W-B-W-B-W-W (one complete octave pattern)
        expected_pattern = ['B', 'W', 'B', 'W', 'B', 'W', 'W', 'B', 'W', 'B', 'W', 'W']
        pattern_start = 34  # Start from F#3 (5 keys left of middle C)
        note_names = ['F#3', 'G3', 'G#3', 'A3', 'A#3', 'B3', 'C4(Middle C)', 'C#4', 'D4', 'D#4', 'E4', 'F4']

        pattern_brightness = []
        for i, expected_type in enumerate(expected_pattern):
            key_index = pattern_start + i
            key_brightness = brightness_values[key_index]
            pattern_brightness.append(key_brightness)

            if expected_type == 'W' and key_brightness < THRESHOLD:
                raise ValueError(f"Middle C verification failed: Expected white key at index {key_index} ({note_names[i]}), but brightness ({key_brightness:.1f}) indicates black key.")
            elif expected_type == 'B' and key_brightness >= THRESHOLD:
                raise ValueError(f"Middle C verification failed: Expected black key at index {key_index} ({note_names[i]}), but brightness ({key_brightness:.1f}) indicates white key.")

        print(f"Middle C verification successful: Index 39 brightness = {middle_c_brightness:.1f}")
        print(f"12-key octave pattern verification successful:")
        for i, note in enumerate(note_names):
            key_type = "White" if expected_pattern[i] == 'W' else "Black"
            print(f"  {note}: {pattern_brightness[i]:.1f} ({key_type})")

        return True

#   def detect_keys_with_verification(self, image: np.ndarray) -> Tuple[List[Tuple[int, int]], List[float]]:
#       """
#       Enhanced detection function that includes middle C verification

#       Args:
#           image: Input keyboard image

#       Returns:
#           Tuple of (key_positions, brightness_values)

#       Raises:
#           ValueError: If middle C verification fails
#       """
#       # Perform standard key detection
#       key_positions, brightness_values = self.detect_keys(image)

#       if not key_positions:
#           return key_positions, brightness_values

#       # Verify middle C positioning
#       try:
#           self.verify_middle_c(brightness_values)
#           print("Key detection and middle C verification completed successfully!")
#       except ValueError as e:
#           print(f"Key detection failed verification: {e}")
#           raise

#       return key_positions, brightness_values

    # def get_key_info(self, key_index: int) -> Optional[Dict]:
    #     """Get detailed information about a specific key"""
    #     if not (0 <= key_index < 88):
    #         return None

    #     key_type = self.key_pattern[key_index]

    #     if key_index < 3:  # A0, A#0, B0
    #         note = self.note_names[9 + key_index]
    #         octave = 0
    #     else:
    #         # C1 and above
    #         adjusted_index = key_index - 3
    #         octave = (adjusted_index // 12) + 1
    #         note = self.note_names[3 + (adjusted_index % 12)]

    #     return {
    #         'index': key_index,
    #         'type': 'white' if key_type == 'W' else 'black',
    #         'note': f"{note}{octave}",
    #         'midi_note': key_index + 21  # A0 = MIDI note 21
    #     }

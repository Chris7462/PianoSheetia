import unittest
import cv2
import numpy as np
import os
import tempfile
from unittest.mock import patch
from PianoSheetia import PianoKeyboard, KeyboardDetector


class TestKeyboardDetector(unittest.TestCase):
    """Test cases for the KeyboardDetector class"""

    def setUp(self):
        """Set up test fixtures with a mock template file"""
        # Create a temporary template file for testing
        self.temp_template = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        # Create a simple test template image (50x20 white rectangle)
        test_template = np.ones((20, 50, 3), dtype=np.uint8) * 255
        cv2.imwrite(self.temp_template.name, test_template)
        self.temp_template.close()

        self.template_path = self.temp_template.name
        self.keyboard = PianoKeyboard()

    def tearDown(self):
        """Clean up temporary files"""
        if os.path.exists(self.temp_template.name):
            os.unlink(self.temp_template.name)

    def test_detector_initialization_valid_template(self):
        """Test detector initialization with valid template"""
        detector = KeyboardDetector(self.template_path)
        self.assertIsNotNone(detector.template)
        self.assertIsNotNone(detector._template_gray)
        self.assertGreater(detector._template_h, 0)
        self.assertGreater(detector._template_w, 0)
        self.assertIsNone(detector.piano_boundary)

    def test_detector_initialization_empty_template_path(self):
        """Test detector initialization with empty template path"""
        with self.assertRaises(ValueError) as context:
            KeyboardDetector("")
        self.assertIn("template_path is required", str(context.exception))

    def test_detector_initialization_nonexistent_template(self):
        """Test detector initialization with non-existent template"""
        with self.assertRaises(FileNotFoundError) as context:
            KeyboardDetector("nonexistent_template.png")
        self.assertIn("Template file not found", str(context.exception))

    def test_detector_constants(self):
        """Test detector class constants"""
        detector = KeyboardDetector(self.template_path)
        self.assertEqual(detector._white_key_y_ratio, 0.75)
        self.assertEqual(detector._black_key_y_ratio, 0.35)
        self.assertEqual(detector._jnd, 5)

    def create_mock_piano_image(self, width=800, height=300):
        """Helper method to create a mock piano image for testing"""
        # Create a simple piano-like image
        image = np.ones((height, width, 3), dtype=np.uint8) * 50  # Dark background

        # Add a white rectangle to simulate piano keys area
        piano_y = height // 4
        piano_height = height // 2
        cv2.rectangle(image, (50, piano_y), (width-50, piano_y + piano_height), (200, 200, 200), -1)

        # Add some variation to simulate keys
        key_width = (width - 100) // 52  # 52 white keys
        for i in range(52):
            x = 50 + i * key_width
            # Alternate brightness slightly
            brightness = 220 if i % 2 == 0 else 180
            cv2.rectangle(image, (x, piano_y + piano_height//2),
                         (x + key_width, piano_y + piano_height), (brightness, brightness, brightness), -1)

        return image

    def test_detect_with_mock_image_success(self):
        """Test successful detection with mock image"""
        detector = KeyboardDetector(self.template_path)
        mock_image = self.create_mock_piano_image()

        # Mock the piano boundary detection to return a valid boundary
        with patch.object(detector, '_detect_piano_boundary') as mock_boundary:
            mock_boundary.return_value = (50, 75, 700, 150)

            # Mock the layout verification to return True
            with patch.object(detector, '_verify_layout') as mock_verify:
                mock_verify.return_value = True

                result = detector.detect(mock_image, self.keyboard)
                self.assertTrue(result)
                self.assertEqual(detector.piano_boundary, (50, 75, 700, 150))

    def test_detect_with_failed_boundary_detection(self):
        """Test detection failure when piano boundary cannot be detected"""
        detector = KeyboardDetector(self.template_path)
        mock_image = self.create_mock_piano_image()

        # Mock boundary detection to fail
        with patch.object(detector, '_detect_piano_boundary') as mock_boundary:
            mock_boundary.return_value = None

            result = detector.detect(mock_image, self.keyboard)
            self.assertFalse(result)
            self.assertIsNone(detector.piano_boundary)

    def test_detect_with_failed_layout_verification(self):
        """Test detection failure when layout verification fails"""
        detector = KeyboardDetector(self.template_path)
        mock_image = self.create_mock_piano_image()

        with patch.object(detector, '_detect_piano_boundary') as mock_boundary:
            mock_boundary.return_value = (50, 75, 700, 150)

            with patch.object(detector, '_calculate_key_positions') as mock_calc_pos:
                mock_calc_pos.return_value = None

                with patch.object(detector, 'sample_brightness') as mock_sample:
                    mock_sample.return_value = None

                    with patch.object(detector, '_verify_layout') as mock_verify:
                        mock_verify.return_value = False

                        result = detector.detect(mock_image, self.keyboard)
                        self.assertFalse(result)

    def test_detect_with_grayscale_image(self):
        """Test detection with grayscale input image"""
        detector = KeyboardDetector(self.template_path)
        color_image = self.create_mock_piano_image()
        gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

        with patch.object(detector, '_detect_piano_boundary') as mock_boundary:
            mock_boundary.return_value = (50, 75, 700, 150)
            with patch.object(detector, '_verify_layout') as mock_verify:
                mock_verify.return_value = True

                result = detector.detect(gray_image, self.keyboard)
                self.assertTrue(result)

    def test_detect_piano_boundary_no_template(self):
        """Test piano boundary detection with no loaded template"""
        detector = KeyboardDetector(self.template_path)
        detector._template_gray = None  # Simulate no template

        mock_image = cv2.cvtColor(self.create_mock_piano_image(), cv2.COLOR_BGR2GRAY)
        result = detector._detect_piano_boundary(mock_image)
        self.assertIsNone(result)

    def test_detect_piano_boundary_low_confidence(self):
        """Test piano boundary detection with low confidence match"""
        detector = KeyboardDetector(self.template_path)

        # Mock cv2.matchTemplate to return low confidence
        with patch('cv2.matchTemplate') as mock_match:
            # Create mock result with low confidence (0.3)
            mock_result = np.array([[0.3]])
            mock_match.return_value = mock_result

            # Mock minMaxLoc to return the low confidence value
            with patch('cv2.minMaxLoc') as mock_minmax:
                mock_minmax.return_value = (0.3, 0.3, (0, 0), (0, 0))

                mock_image = np.ones((300, 800), dtype=np.uint8) * 128
                result = detector._detect_piano_boundary(mock_image, confidence_threshold=0.7)
                self.assertIsNone(result)

    def test_sample_brightness(self):
        """Test brightness sampling functionality"""
        detector = KeyboardDetector(self.template_path)
        detector.piano_boundary = (50, 75, 700, 150)  # Set mock boundary

        # Create a mock grayscale image
        gray_image = np.ones((300, 800), dtype=np.uint8) * 128

        # Set up keyboard with mock positions
        for i, key in enumerate(self.keyboard):
            key.x = 100 + i * 10
            key.y = 150

        # Mock the baseline calculation
        with patch.object(detector, '_calculate_baselines') as mock_baselines:
            mock_baselines.return_value = None

            detector.sample_brightness(gray_image, self.keyboard)

            # Verify all keys have brightness values
            for key in self.keyboard:
                self.assertIsNotNone(key.brightness)
                self.assertIsInstance(key.brightness, int)

            # Verify baseline calculation was called
            mock_baselines.assert_called_once_with(self.keyboard)

    def test_calculate_baselines(self):
        """Test baseline calculation using IQR approach"""
        detector = KeyboardDetector(self.template_path)

        # Set up keyboard with mock brightness values
        white_brightnesses = [200, 210, 205, 195, 215, 190, 220]  # White key brightnesses
        black_brightnesses = [50, 60, 55, 45, 65, 40, 70]        # Black key brightnesses

        white_idx = 0
        black_idx = 0
        for key in self.keyboard:
            if key.color == 'W' and white_idx < len(white_brightnesses):
                key.brightness = white_brightnesses[white_idx]
                white_idx += 1
            elif key.color == 'B' and black_idx < len(black_brightnesses):
                key.brightness = black_brightnesses[black_idx]
                black_idx += 1
            else:
                # Default brightness for remaining keys
                key.brightness = 200 if key.color == 'W' else 50

        detector._calculate_baselines(self.keyboard)

        # Check that baselines were calculated
        self.assertIsNotNone(self.keyboard.white_baseline)
        self.assertIsNotNone(self.keyboard.black_baseline)

        # Check that white baseline is higher than black baseline
        self.assertGreater(self.keyboard.white_baseline, self.keyboard.black_baseline)

    def test_calculate_key_positions(self):
        """Test key position calculation"""
        detector = KeyboardDetector(self.template_path)
        detector.piano_boundary = (50, 75, 700, 150)  # Set mock boundary

        detector._calculate_key_positions(self.keyboard)

        # Verify all keys have been positioned
        for key in self.keyboard:
            self.assertIsNotNone(key.x)
            self.assertIsNotNone(key.y)

        # Verify white keys are positioned correctly
        white_keys = [key for key in self.keyboard if key.color == 'W']
        for i in range(1, len(white_keys)):
            # White keys should be ordered from left to right
            self.assertGreater(white_keys[i].x, white_keys[i-1].x)

        # Verify black keys are positioned between white keys
        for key in self.keyboard:
            if key.color == 'B':
                # Black keys should have y position in upper portion
                self.assertEqual(key.y, 75 + int(150 * 0.35))

    def test_validate_detection_completeness_success(self):
        """Test successful detection completeness validation"""
        detector = KeyboardDetector(self.template_path)

        # Set up keyboard with all keys properly positioned
        for i, key in enumerate(self.keyboard):
            key.x = i * 10
            key.y = 100
            key.brightness = 150

        result = detector._validate_detection_completeness(self.keyboard)
        self.assertTrue(result)

    def test_validate_detection_completeness_missing_position(self):
        """Test detection completeness validation with missing positions"""
        detector = KeyboardDetector(self.template_path)

        # Set up keyboard but leave some positions as None
        for i, key in enumerate(self.keyboard):
            if i < 5:  # First 5 keys missing position
                key.x = None
                key.y = None
                key.brightness = None
            else:
                key.x = i * 10
                key.y = 100
                key.brightness = 150

        result = detector._validate_detection_completeness(self.keyboard)
        self.assertFalse(result)

    def test_validate_detection_completeness_missing_brightness(self):
        """Test detection completeness validation with missing brightness"""
        detector = KeyboardDetector(self.template_path)

        # Set up keyboard but leave some brightness values as None
        for i, key in enumerate(self.keyboard):
            key.x = i * 10
            key.y = 100
            if i < 3:  # First 3 keys missing brightness
                key.brightness = None
            else:
                key.brightness = 150

        result = detector._validate_detection_completeness(self.keyboard)
        self.assertFalse(result)

    def test_validate_baselines_success(self):
        """Test successful baseline validation"""
        detector = KeyboardDetector(self.template_path)

        # Set valid baselines
        self.keyboard.white_baseline = 200
        self.keyboard.black_baseline = 50

        result = detector._validate_baselines(self.keyboard)
        self.assertTrue(result)

    def test_validate_baselines_missing_white_baseline(self):
        """Test baseline validation with missing white baseline"""
        detector = KeyboardDetector(self.template_path)

        self.keyboard.white_baseline = None
        self.keyboard.black_baseline = 50

        result = detector._validate_baselines(self.keyboard)
        self.assertFalse(result)

    def test_validate_baselines_missing_black_baseline(self):
        """Test baseline validation with missing black baseline"""
        detector = KeyboardDetector(self.template_path)

        self.keyboard.white_baseline = 200
        self.keyboard.black_baseline = None

        result = detector._validate_baselines(self.keyboard)
        self.assertFalse(result)

    def test_validate_baselines_out_of_range(self):
        """Test baseline validation with out of range values"""
        detector = KeyboardDetector(self.template_path)

        # Test white baseline out of range
        self.keyboard.white_baseline = 300  # > 255
        self.keyboard.black_baseline = 50

        result = detector._validate_baselines(self.keyboard)
        self.assertFalse(result)

        # Test black baseline out of range
        self.keyboard.white_baseline = 200
        self.keyboard.black_baseline = -10  # < 0

        result = detector._validate_baselines(self.keyboard)
        self.assertFalse(result)

    def test_validate_baselines_inverted_brightness(self):
        """Test baseline validation with inverted brightness (white <= black)"""
        detector = KeyboardDetector(self.template_path)

        self.keyboard.white_baseline = 50   # Should be brighter
        self.keyboard.black_baseline = 200  # Should be darker

        result = detector._validate_baselines(self.keyboard)
        self.assertFalse(result)

    def test_verify_layout_success(self):
        """Test successful layout verification"""
        detector = KeyboardDetector(self.template_path)

        # Set up keyboard with proper values for both completeness and baseline validation
        for i, key in enumerate(self.keyboard):
            key.x = i * 10
            key.y = 100
            # Set brightness based on key type
            if key.color == 'W':
                key.brightness = 200
            else:
                key.brightness = 50

        # Set valid baselines
        self.keyboard.white_baseline = 200
        self.keyboard.black_baseline = 50

        result = detector._verify_layout(self.keyboard)
        self.assertTrue(result)

    def test_verify_layout_fails_completeness(self):
        """Test layout verification failure due to incomplete detection"""
        detector = KeyboardDetector(self.template_path)

        # Set up keyboard but leave some keys incomplete
        for i, key in enumerate(self.keyboard):
            if i < 2:  # First 2 keys incomplete
                key.x = None
                key.y = None
                key.brightness = None
            else:
                key.x = i * 10
                key.y = 100
                key.brightness = 150

        result = detector._verify_layout(self.keyboard)
        self.assertFalse(result)

    def test_verify_layout_fails_baselines(self):
        """Test layout verification failure due to invalid baselines"""
        detector = KeyboardDetector(self.template_path)

        # Set up keyboard with complete detection but invalid baselines
        for i, key in enumerate(self.keyboard):
            key.x = i * 10
            key.y = 100
            key.brightness = 150

        # Set invalid baselines (None)
        self.keyboard.white_baseline = None
        self.keyboard.black_baseline = 50

        result = detector._verify_layout(self.keyboard)
        self.assertFalse(result)

    def test_verify_layout_fails_middle_c(self):
        """Test layout verification failure due to middle C verification failure"""
        detector = KeyboardDetector(self.template_path)

        # Set up keyboard with complete detection and valid baselines
        for i, key in enumerate(self.keyboard):
            key.x = i * 10
            key.y = 100
            # Set appropriate brightness for most keys
            if key.color == 'W':
                key.brightness = 200
            else:
                key.brightness = 50

        # Set wrong brightness for middle C specifically
        middle_c = self.keyboard.find_key_by_name('C4')
        middle_c.brightness = 50  # Too dark for white key

        # Set valid baselines
        self.keyboard.white_baseline = 200
        self.keyboard.black_baseline = 50

        result = detector._verify_layout(self.keyboard)
        self.assertFalse(result)

    def test_verify_middle_c_success(self):
        """Test successful middle C verification"""
        detector = KeyboardDetector(self.template_path)

        # Set up keyboard with appropriate brightness values and baselines
        for i, key in enumerate(self.keyboard):
            key.x = i * 10  # Mock positions
            key.y = 100
            # Set brightness based on key type
            if key.color == 'W':
                key.brightness = 200
            else:
                key.brightness = 50

        # Set baselines
        self.keyboard.white_baseline = 200
        self.keyboard.black_baseline = 50

        result = detector._verify_middle_c(self.keyboard)
        self.assertTrue(result)

    def test_verify_middle_c_failure_wrong_brightness(self):
        """Test middle C verification failure due to wrong brightness"""
        detector = KeyboardDetector(self.template_path)

        # Set up keyboard with appropriate brightness for most keys
        for i, key in enumerate(self.keyboard):
            key.x = i * 10
            key.y = 100
            if key.color == 'W':
                key.brightness = 200
            else:
                key.brightness = 50

        # Set wrong brightness for middle C
        middle_c = self.keyboard.find_key_by_name('C4')
        middle_c.brightness = 50  # Too dark for white key

        # Set baselines
        self.keyboard.white_baseline = 200
        self.keyboard.black_baseline = 50

        result = detector._verify_middle_c(self.keyboard)
        self.assertFalse(result)

    def test_verify_middle_c_pattern_mismatch(self):
        """Test middle C verification failure due to pattern mismatch"""
        detector = KeyboardDetector(self.template_path)

        # Set up keyboard positions
        for i, key in enumerate(self.keyboard):
            key.x = i * 10
            key.y = 100

        # Set baselines
        self.keyboard.white_baseline = 200
        self.keyboard.black_baseline = 50

        # Set wrong brightness pattern around middle C (F#3 to F4)
        f_sharp_3 = self.keyboard.find_key_by_name('F#3')
        pattern_start = f_sharp_3.index

        for i in range(12):  # 12-key pattern
            key_index = pattern_start + i
            key = self.keyboard[key_index]

            # Deliberately set wrong brightness pattern
            if key.color == 'W':
                key.brightness = 50   # Wrong: should be ~200 (bright)
            else:
                key.brightness = 200  # Wrong: should be ~50 (dark)

        # Set correct brightness for other keys
        for i, key in enumerate(self.keyboard):
            if not (pattern_start <= i < pattern_start + 12):
                if key.color == 'W':
                    key.brightness = 200
                else:
                    key.brightness = 50

        result = detector._verify_middle_c(self.keyboard)
        self.assertFalse(result)

    def test_detect_exception_handling(self):
        """Test detection with exception handling"""
        detector = KeyboardDetector(self.template_path)

        # Test with invalid image (None)
        result = detector.detect(None, self.keyboard)
        self.assertFalse(result)

    def test_template_loading_invalid_file(self):
        """Test template loading with invalid image file"""
        # Create a text file instead of image
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("This is not an image")
            invalid_path = f.name

        try:
            detector = KeyboardDetector(invalid_path)
            # Should succeed in initialization but template will be None
            self.assertIsNone(detector.template)
        finally:
            os.unlink(invalid_path)

    def test_piano_boundary_edge_cases(self):
        """Test piano boundary detection edge cases"""
        detector = KeyboardDetector(self.template_path)

        # Test with very small image
        small_image = np.ones((10, 10), dtype=np.uint8)
        result = detector._detect_piano_boundary(small_image)
        self.assertIsNone(result)

        # Test with very large template scale (should be skipped)
        large_image = np.ones((100, 100), dtype=np.uint8)
        result = detector._detect_piano_boundary(large_image)
        # Should still return None due to poor match, but shouldn't crash

    def test_black_key_positioning_magic_numbers(self):
        """Test that black key positioning uses the correct magic number ratios"""
        detector = KeyboardDetector(self.template_path)
        detector.piano_boundary = (50, 75, 700, 150)

        detector._calculate_key_positions(self.keyboard)

        # Test a few specific black key positions to ensure magic numbers are working
        # Find some black keys and verify they're positioned between their white neighbors
        c_sharp_keys = [key for key in self.keyboard if key.name.startswith('C#')]
        for c_sharp in c_sharp_keys:
            # Find adjacent white keys
            left_white = self.keyboard[c_sharp.index - 1]
            right_white = self.keyboard[c_sharp.index + 1]

            self.assertGreater(c_sharp.x, left_white.x)
            self.assertLess(c_sharp.x, right_white.x)

    def test_integration_full_pipeline(self):
        """Integration test for the complete detection pipeline"""
        detector = KeyboardDetector(self.template_path)
        mock_image = self.create_mock_piano_image()

        # Mock only the boundary detection, let everything else run
        with patch.object(detector, '_detect_piano_boundary') as mock_boundary:
            mock_boundary.return_value = (50, 75, 700, 150)

            # This should test the full pipeline:
            # boundary detection -> positioning -> brightness sampling -> baselines -> verification
            result = detector.detect(mock_image, self.keyboard)

            # The result depends on whether the mock image produces good enough
            # brightness patterns for middle C verification
            # At minimum, it shouldn't crash and should return a boolean
            self.assertIsInstance(result, bool)

#   def test_brightness_sampling_with_edge_positions(self):
#       """Test brightness sampling when keys are positioned at image edges"""
#       detector = KeyboardDetector(self.template_path)
#       gray_image = np.ones((100, 100), dtype=np.uint8) * 128

#       # Position some keys at edges where the 5x5 sampling region might go out of bounds
#       edge_positions = [(2, 2), (97, 97), (0, 50), (50, 0)]

#       for i, (x, y) in enumerate(edge_positions):
#           if i < len(self.keyboard):
#               self.keyboard[i].x = x
#               self.keyboard[i].y = y

#       # Mock baseline calculation to avoid issues
#       with patch.object(detector, '_calculate_baselines'):
#           # This should not crash even with edge positions
#           detector.sample_brightness(gray_image, self.keyboard)

#           # Verify brightness was sampled (might be based on smaller regions due to clipping)
#           for i, _ in enumerate(edge_positions):
#               if i < len(self.keyboard):
#                   self.assertIsNotNone(self.keyboard[i].brightness)


if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)

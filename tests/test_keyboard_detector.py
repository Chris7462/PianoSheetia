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
        self.assertEqual(detector._NUM_WHITE_KEYS, 52)
        self.assertEqual(detector._white_key_y_ratio, 0.75)
        self.assertEqual(detector._black_key_y_ratio, 0.35)

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

            with patch.object(detector, '_calculate_key_positions_and_sample_brightness') as mock_calc:
                mock_calc.return_value = True

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

    def test_calculate_key_positions_and_sample_brightness(self):
        """Test key position calculation and brightness sampling"""
        detector = KeyboardDetector(self.template_path)
        detector.piano_boundary = (50, 75, 700, 150)  # Set mock boundary

        mock_image = cv2.cvtColor(self.create_mock_piano_image(), cv2.COLOR_BGR2GRAY)

        result = detector._calculate_key_positions_and_sample_brightness(mock_image, self.keyboard)
        self.assertTrue(result)

        # Verify all keys have been positioned
        for key in self.keyboard:
            self.assertIsNotNone(key.x)
            self.assertIsNotNone(key.y)
            self.assertIsNotNone(key.brightness)

    def test_validate_detection_completeness_success(self):
        """Test successful detection completeness validation"""
        detector = KeyboardDetector(self.template_path)

        # Set up keyboard with all keys properly positioned
        for i, key in enumerate(self.keyboard):
            key.x = i * 10
            key.y = 100
            key.brightness = 150.0

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
                key.brightness = 150.0

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
                key.brightness = 150.0

        result = detector._validate_detection_completeness(self.keyboard)
        self.assertFalse(result)

    def test_verify_layout_success(self):
        """Test successful layout verification"""
        detector = KeyboardDetector(self.template_path)

        # Set up keyboard with proper values for both completeness and middle C verification
        for i, key in enumerate(self.keyboard):
            key.x = i * 10
            key.y = 100
            # Set brightness based on key type
            if key.color == 'W':
                key.brightness = 200.0  # Bright (above threshold)
            else:
                key.brightness = 50.0   # Dark (below threshold)

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
                key.brightness = 150.0

        result = detector._verify_layout(self.keyboard)
        self.assertFalse(result)

    def test_verify_layout_fails_middle_c(self):
        """Test layout verification failure due to middle C verification failure"""
        detector = KeyboardDetector(self.template_path)

        # Set up keyboard with complete detection but wrong middle C brightness
        for i, key in enumerate(self.keyboard):
            key.x = i * 10
            key.y = 100
            if i == 39:  # Middle C with wrong brightness
                key.brightness = 50.0  # Too dark for white key
            elif key.color == 'W':
                key.brightness = 200.0
            else:
                key.brightness = 50.0

        result = detector._verify_layout(self.keyboard)
        self.assertFalse(result)

    def test_verify_middle_c_success(self):
        """Test successful middle C verification"""
        detector = KeyboardDetector(self.template_path)

        # Set up keyboard with mock brightness values
        # Set middle C (index 39) and surrounding keys with appropriate brightness
        for i, key in enumerate(self.keyboard):
            key.x = i * 10  # Mock positions
            key.y = 100
            # Set brightness based on key type (white keys bright, black keys dark)
            if key.color == 'W':
                key.brightness = 200.0  # Bright (above threshold)
            else:
                key.brightness = 50.0   # Dark (below threshold)

        result = detector._verify_middle_c(self.keyboard)
        self.assertTrue(result)

    def test_verify_middle_c_failure_wrong_brightness(self):
        """Test middle C verification failure due to wrong brightness"""
        detector = KeyboardDetector(self.template_path)

        # Set up keyboard but make middle C (index 39) have black key brightness
        for i, key in enumerate(self.keyboard):
            key.x = i * 10
            key.y = 100
            if i == 39:  # Middle C
                key.brightness = 50.0  # Too dark for white key
            elif key.color == 'W':
                key.brightness = 200.0
            else:
                key.brightness = 50.0

        with self.assertRaises(ValueError) as context:
            detector._verify_middle_c(self.keyboard)
        self.assertIn("Middle C verification failed", str(context.exception))

    def test_verify_middle_c_failure_missing_brightness(self):
        """Test middle C verification failure due to missing brightness values"""
        detector = KeyboardDetector(self.template_path)

        # Set up keyboard but leave brightness as None
        for key in self.keyboard:
            key.x = 100
            key.y = 100
            key.brightness = None  # Missing brightness

        with self.assertRaises(ValueError) as context:
            detector._verify_middle_c(self.keyboard)
        self.assertIn("Middle C verification failed", str(context.exception))

    def test_verify_middle_c_pattern_mismatch(self):
        """Test middle C verification failure due to pattern mismatch"""
        detector = KeyboardDetector(self.template_path)

        # Set up keyboard with wrong pattern around middle C
        for i, key in enumerate(self.keyboard):
            key.x = i * 10
            key.y = 100
            # Deliberately set wrong brightness pattern
            if 33 <= i <= 44:  # Pattern around middle C
                # Set opposite brightness (white keys dark, black keys bright)
                if key.color == 'W':
                    key.brightness = 50.0   # Wrong: should be bright
                else:
                    key.brightness = 200.0  # Wrong: should be dark
            else:
                # Set correct brightness for other keys
                if key.color == 'W':
                    key.brightness = 200.0
                else:
                    key.brightness = 50.0

        with self.assertRaises(ValueError) as context:
            detector._verify_middle_c(self.keyboard)
        self.assertIn("Middle C verification failed", str(context.exception))

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

    def test_integration_full_pipeline(self):
        """Integration test for the complete detection pipeline"""
        detector = KeyboardDetector(self.template_path)
        mock_image = self.create_mock_piano_image()

        # Mock only the boundary detection, let everything else run
        with patch.object(detector, '_detect_piano_boundary') as mock_boundary:
            mock_boundary.return_value = (50, 75, 700, 150)

            # This should test the full pipeline:
            # boundary detection -> positioning -> completeness validation -> middle C verification
            result = detector.detect(mock_image, self.keyboard)

            # The result depends on whether the mock image produces good enough
            # brightness patterns for middle C verification
            # At minimum, it shouldn't crash and should return a boolean
            self.assertIsInstance(result, bool)


if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)

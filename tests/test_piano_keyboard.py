import unittest
from PianoSheetia import PianoKeyboard, PianoKey


class TestPianoKey(unittest.TestCase):
    """Test cases for the PianoKey dataclass"""

    def test_piano_key_creation(self):
        """Test creating a PianoKey with all properties"""
        key = PianoKey(index=0, color='W', name='A0', x=100, y=200, brightness=0.5)
        self.assertEqual(key.index, 0)
        self.assertEqual(key.color, 'W')
        self.assertEqual(key.name, 'A0')
        self.assertEqual(key.x, 100)
        self.assertEqual(key.y, 200)
        self.assertEqual(key.brightness, 0.5)

    def test_piano_key_defaults(self):
        """Test creating a PianoKey with default values"""
        key = PianoKey(index=1, color='B', name='A#0')
        self.assertEqual(key.index, 1)
        self.assertEqual(key.color, 'B')
        self.assertEqual(key.name, 'A#0')
        self.assertIsNone(key.x)
        self.assertIsNone(key.y)
        self.assertIsNone(key.brightness)


class TestPianoKeyboard(unittest.TestCase):
    """Test cases for the PianoKeyboard class"""

    def setUp(self):
        """Set up a fresh keyboard for each test"""
        self.keyboard = PianoKeyboard()

    def test_keyboard_initialization(self):
        """Test that keyboard initializes with correct number of keys"""
        self.assertEqual(len(self.keyboard), 88)
        self.assertEqual(len(self.keyboard.keys), 88)

    def test_first_three_keys(self):
        """Test the first three keys (A0, A#0, B0)"""
        self.assertEqual(self.keyboard[0].name, 'A0')
        self.assertEqual(self.keyboard[0].color, 'W')
        self.assertEqual(self.keyboard[0].index, 0)

        self.assertEqual(self.keyboard[1].name, 'A#0')
        self.assertEqual(self.keyboard[1].color, 'B')
        self.assertEqual(self.keyboard[1].index, 1)

        self.assertEqual(self.keyboard[2].name, 'B0')
        self.assertEqual(self.keyboard[2].color, 'W')
        self.assertEqual(self.keyboard[2].index, 2)

    def test_last_key(self):
        """Test the last key (C8)"""
        last_key = self.keyboard[87]
        self.assertEqual(last_key.name, 'C8')
        self.assertEqual(last_key.color, 'W')
        self.assertEqual(last_key.index, 87)

    def test_octave_pattern(self):
        """Test that octave patterns are correct"""
        # Test C1 octave (keys 3-14)
        expected_c1_octave = [
                ('C1', 'W'), ('C#1', 'B'), ('D1', 'W'), ('D#1', 'B'),
                ('E1', 'W'), ('F1', 'W'), ('F#1', 'B'), ('G1', 'W'),
                ('G#1', 'B'), ('A1', 'W'), ('A#1', 'B'), ('B1', 'W')
                ]

        for i, (expected_name, expected_color) in enumerate(expected_c1_octave):
            key = self.keyboard[3 + i]  # C1 starts at index 3
            self.assertEqual(key.name, expected_name)
            self.assertEqual(key.color, expected_color)

    def test_white_keys_count(self):
        """Test that there are 52 white keys"""
        white_keys_count = self.keyboard.white_key_count
        self.assertEqual(white_keys_count, 52)

    def test_black_keys_count(self):
        """Test that there are 36 black keys"""
        black_keys_count = self.keyboard.black_key_count
        self.assertEqual(black_keys_count, 36)

    def test_find_key_by_name(self):
        """Test finding keys by name"""
        # Test existing keys
        c4_key = self.keyboard.find_key_by_name('C4')
        self.assertIsNotNone(c4_key)
        self.assertEqual(c4_key.name, 'C4')
        self.assertEqual(c4_key.color, 'W')

        fs3_key = self.keyboard.find_key_by_name('F#3')
        self.assertIsNotNone(fs3_key)
        self.assertEqual(fs3_key.name, 'F#3')
        self.assertEqual(fs3_key.color, 'B')

        # Test non-existing key
        invalid_key = self.keyboard.find_key_by_name('X9')
        self.assertIsNone(invalid_key)

    def test_getitem_access(self):
        """Test accessing keys using keyboard[index]"""
        # Test valid indices
        key_0 = self.keyboard[0]
        self.assertEqual(key_0.index, 0)

        key_87 = self.keyboard[87]
        self.assertEqual(key_87.index, 87)

    def test_iteration(self):
        """Test iterating over keyboard keys"""
        keys_from_iteration = list(self.keyboard)
        self.assertEqual(len(keys_from_iteration), 88)

        # Check that iteration returns the same keys as direct access
        for i, key in enumerate(self.keyboard):
            self.assertEqual(key, self.keyboard[i])

    def test_repr_method(self):
        """Test the string representation of the keyboard"""
        repr_str = repr(self.keyboard)

        # Test that repr contains expected elements
        self.assertIn("PianoKeyboard with 88 keys:", repr_str)
        self.assertIn("Index", repr_str)
        self.assertIn("Color", repr_str)
        self.assertIn("Name", repr_str)
        self.assertIn("X", repr_str)
        self.assertIn("Y", repr_str)
        self.assertIn("Brightness", repr_str)
        self.assertIn("A0", repr_str)
        self.assertIn("C8", repr_str)

        # Test repr with updated values
        self.keyboard[0].x = 100
        self.keyboard[0].y = 200
        self.keyboard[0].brightness = 0.5
        repr_str_updated = repr(self.keyboard)
        self.assertIn("100", repr_str_updated)
        self.assertIn("200", repr_str_updated)
        self.assertIn("0.500", repr_str_updated)

    def test_key_indices_consistency(self):
        """Test that key indices are consistent with their position in the list"""
        for i, key in enumerate(self.keyboard):
            self.assertEqual(key.index, i)

    def test_specific_notes_exist(self):
        """Test that specific important notes exist"""
        important_notes = ['A0', 'C4', 'A4', 'C8']
        for note in important_notes:
            key = self.keyboard.find_key_by_name(note)
            self.assertIsNotNone(key, f"Note {note} should exist")
            self.assertEqual(key.name, note)

    def test_middle_c_properties(self):
        """Test that middle C (C4) has correct properties"""
        middle_c = self.keyboard.find_key_by_name('C4')
        self.assertIsNotNone(middle_c)
        self.assertEqual(middle_c.color, 'W')
        self.assertEqual(middle_c.name, 'C4')

    def test_all_keys_have_default_none_values(self):
        """Test that all keys start with None values for position and brightness"""
        for key in self.keyboard:
            self.assertIsNone(key.x)
            self.assertIsNone(key.y)
            self.assertIsNone(key.brightness)


if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)

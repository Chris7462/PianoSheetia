from typing import Iterator, List, Optional
from dataclasses import dataclass


@dataclass
class PianoKey:
    """
    Represents a single piano key with immutable properties and mutable position/brightness
    """
    # Immutable properties
    index: int
    type: str  # 'W' or 'B'
    name: str  # 'A0', 'C4', etc.

    # Mutable properties (updated by detector)
    x: Optional[int] = None
    y: Optional[int] = None
    brightness: Optional[float] = None


class PianoKeyboard:
    """
    Represents a complete 88-key piano keyboard with structure and key data
    """

    # Piano structure constants
    TOTAL_KEYS = 88
    OCTAVE_PATTERN = ['W', 'B', 'W', 'B', 'W', 'W', 'B', 'W', 'B', 'W', 'B', 'W']
    NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

    def __init__(self):
        self.keys = self._create_keys()

    def _create_keys(self) -> List[PianoKey]:
        """Create all 88 piano keys efficiently"""
        key_patterns = self._get_key_patterns()
        note_names = self._get_note_names()

        return [
            PianoKey(index=i, type=key_patterns[i], name=note_names[i])
            for i in range(self.TOTAL_KEYS)
        ]

    def _get_key_patterns(self) -> List[str]:
        """Generate the complete 88-key pattern efficiently"""
        # Start with A0, A#0, B0
        patterns = ['W', 'B', 'W']

        # Add 7 complete octaves (84 keys: 12 * 7)
        patterns.extend(self.OCTAVE_PATTERN * 7)

        # Add final C8
        patterns.append('W')

        return patterns

    def _get_note_names(self) -> List[str]:
        """Generate note names for all 88 keys efficiently"""
        # Start with A0, A#0, B0
        names = ['A0', 'A#0', 'B0']

        # Add 7 complete octaves
        for octave in range(1, 8):
            names.extend(f"{note}{octave}" for note in self.NOTE_NAMES)

        # Add final C8
        names.append('C8')

        return names

    def get_white_keys(self) -> List[PianoKey]:
        """Return all white keys"""
        return [key for key in self.keys if key.type == 'W']

    def get_black_keys(self) -> List[PianoKey]:
        """Return all black keys"""
        return [key for key in self.keys if key.type == 'B']

    def find_key_by_name(self, name: str) -> Optional[PianoKey]:
        """Find a key by its note name (e.g., 'C4', 'F#3')"""
        for key in self.keys:
            if key.name == name:
                return key
        return None

    #   def get_octave_keys(self, octave: int) -> List[PianoKey]:
    #       """Get all keys in a specific octave"""
    #       if octave < 0 or octave > 8:
    #           raise ValueError(f"Octave must be between 0 and 8, got {octave}")

    #       return [key for key in self.keys if key.name.endswith(str(octave))]

    #   def get_keys_in_range(self, start_key: str, end_key: str) -> List[PianoKey]:
    #       """Get keys within a range (inclusive)"""
    #       start_idx = next((key.index for key in self.keys if key.name == start_key), None)
    #       end_idx = next((key.index for key in self.keys if key.name == end_key), None)

    #       if start_idx is None or end_idx is None:
    #           raise ValueError(f"Invalid key names: {start_key} or {end_key}")

    #       if start_idx > end_idx:
    #           start_idx, end_idx = end_idx, start_idx

    #       return self.keys[start_idx:end_idx + 1]

    def update_key_position(self, index: int, x: int, y: int) -> None:
        """Update a key's position"""
        if not (0 <= index < self.TOTAL_KEYS):
            raise IndexError(f"Key index {index} out of range (0-{self.TOTAL_KEYS-1})")

        self.keys[index].x = x
        self.keys[index].y = y

    def update_key_brightness(self, index: int, brightness: float) -> None:
        """Update a key's brightness"""
        if not (0 <= index < self.TOTAL_KEYS):
            raise IndexError(f"Key index {index} out of range (0-{self.TOTAL_KEYS-1})")

        self.keys[index].brightness = brightness

    def __getitem__(self, index: int) -> PianoKey:
        """Allow keyboard[i] access to keys"""
        if not (0 <= index < self.TOTAL_KEYS):
            raise IndexError(f"Key index {index} out of range (0-{self.TOTAL_KEYS-1})")
        return self.keys[index]

    def __len__(self) -> int:
        """Return number of keys"""
        return self.TOTAL_KEYS

    def __iter__(self) -> Iterator[PianoKey]:
        """Allow iteration over keys"""
        return iter(self.keys)

    def __repr__(self) -> str:
        """Return a detailed string representation of the piano keyboard"""
        lines = [f"PianoKeyboard with {self.TOTAL_KEYS} keys:"]
        lines.append("-" * 60)
        lines.append(f"{'Index':<5} {'Type':<4} {'Name':<6} {'X':<8} {'Y':<8} {'Brightness'}")
        lines.append("-" * 60)

        for key in self.keys:
            x_str = str(key.x) if key.x is not None else "None"
            y_str = str(key.y) if key.y is not None else "None"
            brightness_str = f"{key.brightness:.3f}" if key.brightness is not None else "None"

            lines.append(f"{key.index:<5} {key.type:<4} {key.name:<6} {x_str:<8} {y_str:<8} {brightness_str}")

        return "\n".join(lines)

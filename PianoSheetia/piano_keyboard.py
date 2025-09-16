"""
piano_keyboard.py

Defines the PianoKey dataclass and PianoKeyboard class for representing
an 88-key piano, including note names, key types (white/black),
and mutable properties like position and brightness.
"""

from typing import Iterator, List, Optional
from dataclasses import dataclass


@dataclass
class PianoKey:
    """Represents a single piano key with immutable properties and mutable position/brightness"""
    # Immutable properties
    index: int
    type: str  # 'W' or 'B'
    name: str  # 'A0', 'C4', etc.

    # Mutable properties (updated by detector)
    x: Optional[int] = None
    y: Optional[int] = None
    brightness: Optional[float] = None
    default_brightness: Optional[float] = None  # Baseline brightness for comparison


class PianoKeyboard:
    """Represents a complete 88-key piano keyboard with structure and key data"""

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

    def update_position(self, index: int, x: int, y: int) -> None:
        """Update a key's position"""
        self.keys[index].x = x
        self.keys[index].y = y

    def update_brightness(self, index: int, brightness: float) -> None:
        """Update a key's brightness"""
        self.keys[index].brightness = brightness

    def get_key_colors(self) -> List[str]:
        """Return list of key colors for compatibility with detector"""
        return [key.type for key in self.keys]

    def __getitem__(self, index: int) -> PianoKey:
        """Allow keyboard[i] access to keys"""
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
        lines.append('-' * 80)
        lines.append(f'{"Index":<5} {"Type":<4} {"Name":<6} {"X":<8} {"Y":<8} {"Brightness":<12} {"Default"}')
        lines.append('-' * 80)

        for key in self.keys:
            x_str = str(key.x) if key.x is not None else "None"
            y_str = str(key.y) if key.y is not None else "None"
            brightness_str = f"{key.brightness:.3f}" if key.brightness is not None else "None"
            default_str = f"{key.default_brightness:.3f}" if key.default_brightness is not None else "None"

            lines.append(f"{key.index:<5} {key.type:<4} {key.name:<6}"
                         f"{x_str:<8} {y_str:<8} {brightness_str:<12} {default_str}")

        return "\n".join(lines)

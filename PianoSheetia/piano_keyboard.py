"""
piano_keyboard.py

Defines the PianoKey dataclass and PianoKeyboard class for representing
an 88-key piano, including note names, key colors (white/black),
and mutable properties like position and brightness.
"""

from typing import Final, Iterator, List, Optional
from dataclasses import dataclass


@dataclass
class PianoKey:
    """Represents a single piano key with immutable properties and mutable position/brightness"""
    # Immutable properties
    index: int
    color: str  # 'W' or 'B'
    name: str  # 'A0', 'C4', etc.

    # Mutable properties (updated by detector)
    x: Optional[int] = None
    y: Optional[int] = None
    brightness: Optional[int] = None


class PianoKeyboard:
    """Represents a complete 88-key piano keyboard with structure and key data"""

    # Piano structure constants
    _total_keys: Final[int] = 88

    white_baseline: int = None
    black_baseline: int = None

    def __init__(self):
        self.keys = self._create_keys()

    @property
    def white_key_count(self) -> int:
        """Number of white keys on the piano"""
        return sum(1 for key in self.keys if key.color == 'W')

    @property
    def black_key_count(self) -> int:
        """Number of black keys on the piano"""
        return self._total_keys - self.white_key_count

    def find_key_by_name(self, name: str) -> Optional[PianoKey]:
        """Find a key by its note name (e.g., 'C4', 'F#3')"""
        for key in self.keys:
            if key.name == name:
                return key
        return None

    def get_key_colors(self) -> List[str]:
        """Return list of key colors for compatibility with detector"""
        return [key.color for key in self.keys]

    def _create_keys(self) -> List[PianoKey]:
        """Create all 88 piano keys efficiently"""
        key_colors = self._generate_key_color_pattern()
        note_names = self._generate_key_note_names()

        return [
            PianoKey(index=i, color=key_colors[i], name=note_names[i])
            for i in range(self._total_keys)
        ]

    def _generate_key_color_pattern(self) -> List[str]:
        """Generate the complete 88-key color pattern efficiently"""
        # Start with A0, A#0, B0
        colors = ['W', 'B', 'W']

        # Add 7 complete octaves (84 keys: 12 * 7)
        octave_color_pattern = ['W', 'B', 'W', 'B', 'W', 'W', 'B', 'W', 'B', 'W', 'B', 'W']
        colors.extend(octave_color_pattern * 7)

        # Add final C8
        colors.append('W')

        return colors

    def _generate_key_note_names(self) -> List[str]:
        """Generate note names for all 88 keys efficiently"""
        # Start with A0, A#0, B0
        names = ['A0', 'A#0', 'B0']

        # Add 7 complete octaves
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        for octave in range(1, 8):
            names.extend(f"{note}{octave}" for note in note_names)

        # Add final C8
        names.append('C8')

        return names

    def __getitem__(self, index: int) -> PianoKey:
        """Allow keyboard[i] access to keys"""
        return self.keys[index]

    def __len__(self) -> int:
        """Return number of keys"""
        return self._total_keys

    def __iter__(self) -> Iterator[PianoKey]:
        """Allow iteration over keys"""
        return iter(self.keys)

    def __repr__(self) -> str:
        """Return a detailed string representation of the piano keyboard"""
        lines = [f"PianoKeyboard with {self._total_keys} keys:"]
        lines.append('-' * 50)
        lines.append(f'{"Index":<5} {"Color":<5} {"Name":<5} {"X":<8} {"Y":<8} {"Brightness":<8}')
        lines.append('-' * 50)

        for key in self.keys:
            x_str = str(key.x) if key.x is not None else "None"
            y_str = str(key.y) if key.y is not None else "None"
            brightness_str = str(key.brightness) if key.brightness is not None else "None"

            lines.append(f"{key.index:<5} {key.color:<5} {key.name:<6}"
                         f"{x_str:<8} {y_str:<8} {brightness_str:<8}")

        white_baseline_str = (
            f"{self.white_baseline}"
            if self.white_baseline is not None
            else "None"
        )

        lines.append(f"\nWhite baseline: {white_baseline_str}")

        black_baseline_str = (
            f"{self.black_baseline}"
            if self.black_baseline is not None
            else "None"
        )

        lines.append(f"Black baseline: {black_baseline_str}")

        return "\n".join(lines)

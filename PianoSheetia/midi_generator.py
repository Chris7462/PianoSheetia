"""
midi_generator.py

Provides the MidiGenerator class for handling MIDI file creation and event generation
from piano key state changes.
"""

from mido import Message, MidiFile, MidiTrack
from typing import List, Optional


class MidiGenerator:
    """
    Handles MIDI file creation and event generation from piano key states.

    Maintains internal state for tracking key changes and generating appropriate
    MIDI note on/off events with proper timing.
    """

    # MIDI constants
    MIDI_VELOCITY_ON = 64
    MIDI_VELOCITY_OFF = 127
    MIDI_TICKS_PER_BEAT = 480

    def __init__(self, fps: float):
        """
        Initialize the MIDI generator with a new MIDI file and track.

        Args:
            fps: Video frames per second for timing calculations
        """
        self.fps = fps
        self.midi_file = MidiFile(ticks_per_beat=self.MIDI_TICKS_PER_BEAT)
        self.track = MidiTrack()
        self.midi_file.tracks.append(self.track)

        self.previous_key_states: Optional[List[int]] = None
        self.frame_count = 0
        self.last_mod = 0  # Track when last MIDI event was generated

    def process_frame(self, current_key_states: List[int]) -> None:
        """
        Process a frame's key states and generate MIDI events for state changes.

        Args:
            current_key_states: List of 0/1 values indicating key press states
        """
        # Initialize previous states on first frame
        if self.previous_key_states is None:
            self.previous_key_states = [0] * len(current_key_states)

        # Generate MIDI events for key state changes
        for i, (current_state, previous_state) in enumerate(zip(current_key_states, self.previous_key_states)):
            if current_state != previous_state:
                midi_note = i + 21  # A0 = 21, so key index + 21

                # Calculate timing - matches original logic
                if self.last_mod == 0 and self.frame_count > self.fps:
                    self.last_mod = self.frame_count - self.fps

                time_delta = int((self.frame_count - self.last_mod) * (self.MIDI_TICKS_PER_BEAT / self.fps))

                if current_state == 1:
                    # Note on
                    event = Message('note_on', note=midi_note,
                                  velocity=self.MIDI_VELOCITY_ON,
                                  time=time_delta)
                else:
                    # Note off
                    event = Message('note_off', note=midi_note,
                                  velocity=self.MIDI_VELOCITY_OFF,
                                  time=time_delta)

                self.track.append(event)
                self.last_mod = self.frame_count

        # Update state for next frame
        self.previous_key_states = current_key_states.copy()
        self.frame_count += 1

    def save(self, output_path: str) -> bool:
        """
        Save the MIDI file to the specified path.

        Args:
            output_path: Path where the MIDI file should be saved

        Returns:
            bool: True if save was successful, False otherwise
        """
        try:
            self.midi_file.save(output_path)
            return True
        except Exception as e:
            print(f"Error saving MIDI file: {e}")
            return False

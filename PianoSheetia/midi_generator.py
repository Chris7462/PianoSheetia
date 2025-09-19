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
    _midi_velocity_on = 64
    _midi_velocity_off = 127
    _midi_ticks_per_beat = 480

    def __init__(self, fps: float):
        """
        Initialize the MIDI generator with a new MIDI file and track.

        Args:
            fps: Video frames per second for timing calculations
        """
        self.fps = fps
        self.midi_file = MidiFile(ticks_per_beat=self._midi_ticks_per_beat)
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
        events_in_this_frame = []
        for i, (current_state, previous_state) in enumerate(zip(current_key_states, self.previous_key_states)):
            if current_state != previous_state:
                midi_note = i + 21  # A0 = 21, so key index + 21

                if current_state == 1:
                    # Note on
                    event = Message('note_on', note=midi_note,
                                  velocity=self._midi_velocity_on,
                                  time=0)  # Will set time for first event only
                else:
                    # Note off
                    event = Message('note_off', note=midi_note,
                                  velocity=self._midi_velocity_off,
                                  time=0)  # Will set time for first event only

                events_in_this_frame.append(event)

        # Add events to track with proper timing
        if events_in_this_frame:
            # Calculate timing - matches original logic
            if self.last_mod == 0 and self.frame_count > self.fps:
                self.last_mod = self.frame_count - self.fps

            time_delta = int((self.frame_count - self.last_mod) * (self._midi_ticks_per_beat / self.fps))

            # First event gets the time delta, rest get 0
            events_in_this_frame[0].time = time_delta

            # Add all events to track
            for event in events_in_this_frame:
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

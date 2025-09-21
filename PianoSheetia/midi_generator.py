"""
midi_generator.py

Provides the MidiGenerator class for handling MIDI file creation and event generation
from piano key state changes with dynamic velocity calculation and tempo detection.
"""

from mido import Message, MidiFile, MidiTrack, MetaMessage
from typing import List, Optional, Tuple
import numpy as np
from collections import defaultdict


class MidiGenerator:
    """
    Handles MIDI file creation and event generation from piano key states.

    Maintains internal state for tracking key changes and generating appropriate
    MIDI note on/off events with proper timing, dynamic velocity, and auto-detected tempo.
    """

    # MIDI constants
    _midi_velocity_off = 64
    _base_ticks_per_beat = 480

    def __init__(self, fps: float, dynamic_velocity: bool = True):
        """
        Initialize the MIDI generator with a new MIDI file and track.

        Args:
            fps: Video frames per second for timing calculations
            dynamic_velocity: Whether to calculate velocity from brightness changes
        """
        self.fps = fps
        self.dynamic_velocity = dynamic_velocity

        # Scale ticks_per_beat based on FPS for better timing resolution
        self._midi_ticks_per_beat = int(self._base_ticks_per_beat * (fps / 30.0))

        self.midi_file = MidiFile(ticks_per_beat=self._midi_ticks_per_beat)
        self.track = MidiTrack()
        self.midi_file.tracks.append(self.track)

        # State tracking
        self.previous_key_states: Optional[List[int]] = None
        self.previous_key_brightness: Optional[List[int]] = None
        self.frame_count = 0
        self.last_mod = 0  # Track when last MIDI event was generated

        # Video-synced tempo: Set tempo so that MIDI time matches video time exactly
        # Use 60 BPM so that 1 beat = 1 second, making timing calculations straightforward
        self.sync_bpm = 60
        self.tempo_set = False

    def process_frame(self, current_key_states: List[int], key_brightness: Optional[List[int]] = None) -> None:
        """
        Process a frame's key states and generate MIDI events for state changes.

        Args:
            current_key_states: List of 0/1 values indicating key press states
            key_brightness: Optional list of current brightness values for velocity calculation
        """
        # Initialize previous states on first frame
        if self.previous_key_states is None:
            self.previous_key_states = [0] * len(current_key_states)
            if key_brightness is not None:
                self.previous_key_brightness = key_brightness.copy()

        # Generate MIDI events for key state changes
        events_in_this_frame = []

        for i, (current_state, previous_state) in enumerate(zip(current_key_states, self.previous_key_states)):
            if current_state != previous_state:
                midi_note = i + 21  # A0 = 21, so key index + 21

                if current_state == 1:
                    # Note on - calculate velocity from brightness change
                    velocity = self._calculate_velocity(i, key_brightness)
                    event = Message('note_on', note=midi_note,
                                    velocity=velocity,
                                    time=0)  # Will set time for first event only

                else:
                    # Note off - use fixed release velocity
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
        if key_brightness is not None:
            self.previous_key_brightness = key_brightness.copy()
        self.frame_count += 1

    def _calculate_velocity(self, key_index: int, key_brightness: Optional[List[int]]) -> int:
        """
        Calculate MIDI velocity based on brightness changes.

        Args:
            key_index: Index of the key being pressed
            key_brightness: Current brightness values for all keys

        Returns:
            MIDI velocity value (1-127)
        """
        if not self.dynamic_velocity or key_brightness is None or self.previous_key_brightness is None:
            # Use default medium velocity if dynamic calculation unavailable
            return 64

        # Calculate brightness change for this key
        current_brightness = key_brightness[key_index]
        previous_brightness = self.previous_key_brightness[key_index]
        brightness_change = abs(current_brightness - previous_brightness)

        # Map brightness change to velocity
        # Assume brightness changes of 0-100 map to velocities 20-127
        # This gives a reasonable range while avoiding velocity 0 (note off)
        min_velocity = 20
        max_velocity = 127
        max_brightness_change = 100

        # Linear mapping with clipping
        velocity_range = max_velocity - min_velocity
        normalized_change = min(brightness_change / max_brightness_change, 1.0)
        velocity = int(min_velocity + (normalized_change * velocity_range))

        # Ensure velocity is in valid range
        velocity = max(1, min(127, velocity))

        return velocity

    def _add_tempo_to_track(self) -> None:
        """Add video-synced tempo information to the MIDI track."""
        if not self.tempo_set:
            # Set tempo to 60 BPM so 1 beat = 1 second
            # This makes MIDI timing match video timing exactly
            microseconds_per_beat = int(60_000_000 / self.sync_bpm)  # 60 BPM = 1,000,000 microseconds per beat
            tempo_msg = MetaMessage('set_tempo', tempo=microseconds_per_beat, time=0)

            # Insert tempo message at the beginning of the track
            self.track.insert(0, tempo_msg)
            self.tempo_set = True

            print(f"Added video-synced tempo: {self.sync_bpm} BPM (1 beat = 1 second)")


    def get_velocity_statistics(self) -> Tuple[int, int, float]:
        """
        Get statistics about velocities used in this MIDI file.

        Returns:
            Tuple of (min_velocity, max_velocity, avg_velocity)
        """
        velocities = []
        for message in self.track:
            if message.type == 'note_on' and message.velocity > 0:
                velocities.append(message.velocity)

        if not velocities:
            return 0, 0, 0.0

        return min(velocities), max(velocities), np.mean(velocities)

    def save(self, output_path: str) -> bool:
        """
        Save the MIDI file to the specified path.

        Args:
            output_path: Path where the MIDI file should be saved

        Returns:
            bool: True if save was successful, False otherwise
        """
        try:
            # Add tempo information before saving
            self._add_tempo_to_track()

            self.midi_file.save(output_path)

            # Print statistics
            if self.dynamic_velocity:
                min_vel, max_vel, avg_vel = self.get_velocity_statistics()
                print(f"MIDI velocity statistics: min={min_vel}, max={max_vel}, avg={avg_vel:.1f}")

            print(f"MIDI saved with video-synced tempo: {self.sync_bpm} BPM")
            return True
        except Exception as e:
            print(f"Error saving MIDI file: {e}")
            return False

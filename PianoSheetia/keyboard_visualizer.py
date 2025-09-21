"""
visualization.py

Provides visualization utilities for keyboard detection and piano key analysis.
"""

import cv2
import numpy as np
from typing import Optional, Tuple

from .piano_keyboard import PianoKeyboard


def create_detection_visualization(
        image: np.ndarray,
        keyboard: PianoKeyboard,
        piano_boundary: Optional[Tuple[int, int, int, int]] = None,
        output_path: str = "output/keyboard_detection.jpg"
    ) -> bool:
    """
    Create and save a visualization of keyboard detection results.

    Args:
        image: Original image with detected keyboard
        keyboard: PianoKeyboard object with detected key positions
        piano_boundary: Optional tuple (x, y, width, height) of piano boundary
        output_path: Path where visualization image will be saved

    Returns:
        bool: True if visualization was created successfully, False otherwise
    """
    try:
        # Create a copy of the image to draw on
        vis_image = image.copy()

        # Draw piano boundary if provided
        if piano_boundary:
            x, y, w, h = piano_boundary
            cv2.rectangle(vis_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # Add boundary label
            cv2.putText(vis_image, "Piano Boundary", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Draw key positions
        white_key_count = 0
        black_key_count = 0

        for key in keyboard:
            if key.x is not None and key.y is not None:
                if key.color == 'W':
                    # White keys: white circle with green outline
                    cv2.circle(vis_image, (key.x, key.y), 1, (255, 255, 255), -1)
                    cv2.circle(vis_image, (key.x, key.y), 5, (0, 255, 0), 1)
                    white_key_count += 1
                else:
                    # Black keys: black circle with green outline
                    cv2.circle(vis_image, (key.x, key.y), 1, (0, 0, 0), -1)
                    cv2.circle(vis_image, (key.x, key.y), 5, (0, 255, 0), 1)
                    black_key_count += 1

        # Add statistics text
        stats_text = f"Keys detected: {white_key_count + black_key_count}/88 (W:{white_key_count}, B:{black_key_count})"
        cv2.putText(vis_image, stats_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Save the visualization
        success = cv2.imwrite(output_path, vis_image)
        if success:
            print(f"Keyboard detection visualization saved as '{output_path}'")
            return True
        else:
            print(f"Failed to save visualization to '{output_path}'")
            return False

    except Exception as e:
        print(f"Error creating detection visualization: {e}")
        return False

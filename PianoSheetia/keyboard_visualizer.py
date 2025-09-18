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

def create_brightness_visualization(
        image: np.ndarray,
        keyboard: PianoKeyboard,
        output_path: str = "output/brightness_visualization.jpg"
    ) -> bool:
    """
    Create a visualization showing brightness values at key positions.

    Args:
        image: Current frame image
        keyboard: PianoKeyboard object with brightness values
        output_path: Path where visualization image will be saved

    Returns:
        bool: True if visualization was created successfully, False otherwise
    """
    try:
        vis_image = image.copy()

        for key in keyboard:
            if key.x is not None and key.y is not None and key.brightness is not None:
                # Choose color based on brightness (darker = more pressed)
                brightness_normalized = min(255, max(0, int(key.brightness)))
                color = (brightness_normalized, brightness_normalized, brightness_normalized)

                # Draw circle with brightness-based color
                cv2.circle(vis_image, (key.x, key.y), 5, color, -1)
                cv2.circle(vis_image, (key.x, key.y), 7, (0, 255, 0), 1)

                # Add brightness value as text for some keys (every 12th key to avoid clutter)
                if key.index % 12 == 0:
                    brightness_text = f"{key.brightness:.0f}"
                    cv2.putText(vis_image, brightness_text,
                                (key.x - 15, key.y - 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Save the visualization
        success = cv2.imwrite(output_path, vis_image)
        if success:
            print(f"Brightness visualization saved as '{output_path}'")
            return True
        else:
            print(f"Failed to save brightness visualization to '{output_path}'")
            return False

    except Exception as e:
        print(f"Error creating brightness visualization: {e}")
        return False

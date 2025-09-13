import cv2
import numpy as np
from typing import Dict
import sys

from PianoSheetia import PianoKeyDetector


def create_debug_visualization(detector: PianoKeyDetector, image: np.ndarray, debug_info: Dict):
    """Create comprehensive debug visualization"""
    vis_image = image.copy()

    # Draw piano boundary
    if debug_info.get('piano_boundary'):
        x, y, w, h = debug_info['piano_boundary']
        cv2.rectangle(vis_image, (x, y), (x + w, y + h), (0, 255, 255), 3)
        cv2.putText(vis_image, 'Piano Boundary', (x, y-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    # Draw detected keys
    if debug_info.get('key_positions'):
        positions = debug_info['key_positions']

        for i, (x, y) in enumerate(positions):
            if i < len(detector.key_pattern):
                key_type = detector.key_pattern[i]

                if key_type == 'W':
                    # White keys - blue circles with white border
                    cv2.circle(vis_image, (x, y), 6, (255, 0, 0), -1)
                    cv2.circle(vis_image, (x, y), 9, (255, 255, 255), 2)
                else:
                    # Black keys - red circles with white border
                    cv2.circle(vis_image, (x, y), 4, (0, 0, 255), -1)
                    cv2.circle(vis_image, (x, y), 7, (255, 255, 255), 2)

                # Add key index
                cv2.putText(vis_image, str(i), (x-8, y-15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)

    # Add summary information
    info_text = [
        f"Keys detected: {len(debug_info.get('key_positions', []))}",
        f"White keys: {len(detector.white_key_indices)}",
        f"Black keys: {len(detector.black_key_indices)}"
    ]

    for i, text in enumerate(info_text):
        cv2.putText(vis_image, text, (10, 30 + i*25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Save visualization
    output_path = 'piano_detection_result.jpg'
    cv2.imwrite(output_path, vis_image)
    print(f"Debug visualization saved as '{output_path}'")


def detect_piano_keys(image_path: str, template_path: str):
    """Detect piano keys from a single image"""
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return None, None

    # Create detector and detect keys
    detector = PianoKeyDetector(template_path)
    key_positions = detector.detect_keys(image)

    # Create debug info for visualization
    piano_boundary = detector.detect_piano_boundary(image)
    debug_info = {
        'piano_boundary': piano_boundary,
        'key_positions': key_positions
    }

    # Create debug visualization
    create_debug_visualization(detector, image, debug_info)

    # Print results
    print(f"\nDetection Results:")
    print(f"Total keys detected: {len(key_positions)}")

    # Show first few keys as examples
    for i in range(min(10, len(key_positions))):
        x, y = key_positions[i]
        key_type = 'white' if detector.key_pattern[i] == 'W' else 'black'
        print(f"Key {i:2d}: {key_type:5s} at ({x:3d}, {y:3d})")

    if len(key_positions) > 10:
        print(f"... and {len(key_positions) - 10} more keys")

    return key_positions, detector


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python test_piano_detector.py <image_path> <template_path>")
        print("Example: python test_piano_detector.py keyboard.jpg ./template/piano-88-keys.png")
        sys.exit(1)

    # image_path = sys.argv[1]
    # template_path = sys.argv[2]
    image_path = './data/videos/Interstellar/out0018.png'
    # image_path = './videos/Reverie/out0018.png'
    template_path = './data/template/piano-88-keys-0_5.png'

    detect_piano_keys(image_path, template_path)

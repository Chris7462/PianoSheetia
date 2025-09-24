import cv2
import sys
import os


def get_first_valid_frame(video):
    """Get the first non-black frame from video"""
    for i in range(10):  # Try first 10 frames
        ret, frame = video.read()
        if not ret:
            return None, None

        # Check if frame is not completely black
        mean_brightness = cv2.mean(frame)[0]
        if mean_brightness > 1.0:
            if i > 0:
                print(f"Using frame {i+1} for region selection (frame 1 was black)")
            return ret, frame

    return None, None


if __name__ == "__main__":
    # Check command line arguments
    if len(sys.argv) != 2:
        print("Usage: python template_generator.py <video_path>")
        print("This script helps you select the keyboard region in your piano video.")
        sys.exit(1)

    video_path = sys.argv[1]

    # Check if video exists
    if not os.path.exists(video_path):
        print(f"Error: Video file '{video_path}' not found")
        sys.exit(1)

    # Load video
    try:
        video = cv2.VideoCapture(video_path)
        if not video.isOpened():
            print(f"Error: Could not open video file '{video_path}'")
            sys.exit(1)
        print(f"Loaded video: {video_path}")
    except Exception as e:
        print(f"Error loading video: {e}")
        sys.exit(1)

    # Get first valid frame
    ret, frame = get_first_valid_frame(video)
    if not ret or frame is None:
        print("Error: Could not read any valid frames from video")
        sys.exit(1)

    print("Found valid frame for region selection")

    # Instructions for user
    print("\n" + "="*60)
    print("KEYBOARD REGION SELECTION")
    print("="*60)
    print("Instructions:")
    print("1. A window will open showing the video frame")
    print("2. Click and drag to select the ENTIRE 88-key keyboard region")
    print("3. Make sure to include all keys from leftmost to rightmost")
    print("4. Press SPACE or ENTER to confirm selection")
    print("5. Press ESC to cancel")
    print("="*60)

    # Create window for region selection
    cv2.namedWindow("Select Keyboard Region", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Select Keyboard Region", 1200, 800)

    # Let user select region
    print("\nSelecting region... (click and drag to select keyboard area)")
    region = cv2.selectROI("Select Keyboard Region", frame, showCrosshair=True)

    # Extract coordinates
    x, y, w, h = region

    # Check if user cancelled (selectROI returns (0,0,0,0) on cancel)
    if w == 0 or h == 0:
        print("Region selection cancelled")
        cv2.destroyAllWindows()
        video.release()
        sys.exit(0)

    # Convert to corner coordinates for display
    x1, y1 = x, y
    x2, y2 = x + w, y + h

    print(f"Selected region: ({x1}, {y1}) to ({x2}, {y2})")
    print(f"Region size: {w} x {h} pixels")

    # Extract the ROI from the frame
    roi = frame[y:y+h, x:x+w]

    # Resize ROI to half its original size
    new_width = w // 2
    new_height = h // 2
    resized_roi = cv2.resize(roi, (new_width, new_height), interpolation=cv2.INTER_AREA)

    print(f"Resized ROI to: {new_width} x {new_height} pixels")

    # Save the resized ROI as PNG
    output_file = "keyboard_region.png"
    try:
        success = cv2.imwrite(output_file, resized_roi)
        if success:
            print(f"\nROI saved to '{output_file}'")
            print("The selected keyboard region has been extracted and resized to half size!")
        else:
            print(f"Error: Failed to save image to '{output_file}'")
            sys.exit(1)

        # Show confirmation with original selection
        confirmation_frame = frame.copy()
        cv2.rectangle(confirmation_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(confirmation_frame, "Selected Keyboard Region",
                    (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Selected Keyboard Region", confirmation_frame)

        # Also show the extracted and resized ROI
        cv2.namedWindow("Extracted ROI (Half Size)", cv2.WINDOW_NORMAL)
        cv2.imshow("Extracted ROI (Half Size)", resized_roi)

        print("\nPress any key to close...")
        cv2.waitKey(0)

    except Exception as e:
        print(f"Error processing ROI: {e}")
        sys.exit(1)

    # Cleanup
    cv2.destroyAllWindows()
    video.release()

import sys
import argparse
import cv2
import numpy as np
from mido import Message, MidiFile, MidiTrack, MetaMessage
import os
from PianoSheetia import VideoDownloader, KeyboardDetector, PianoKeyboard


def get_pressed_keys(keyboard, activation_threshold):
    """Determine which keys are currently pressed based on brightness changes"""
    pressed = []
    for key in keyboard:
        if key.brightness is None or key.default_brightness is None:
            pressed.append(0)
        elif abs(key.brightness - key.default_brightness) > activation_threshold:
            pressed.append(1)
        else:
            pressed.append(0)
    return pressed

def sample_current_brightness(image, keyboard):
    """Sample current brightness values for all keys"""
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image

    # Sample brightness at each key position
    for key in keyboard:
        if key.x is not None and key.y is not None:
            # Ensure coordinates are within image bounds
            y = min(max(0, key.y), gray_image.shape[0] - 1)
            x = min(max(0, key.x), gray_image.shape[1] - 1)
            key.brightness = float(gray_image[y, x])

def convert(video, output="out.mid", threshold=30, template_path="data/template/piano-88-keys-0_5.png"):
    """Main conversion function"""
    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)

    # Create VideoDownloader instance
    video_downloader = VideoDownloader()

    # Get video file (handles both URLs and local files)
    input_video = video_downloader.get_video_file(video)
    if input_video is None:
        print("Failed to get video file")
        sys.exit(1)

    # Validate template file
    if not os.path.exists(template_path):
        print(f"Template file not found: {template_path}")
        print("Please ensure template file exists or specify correct path")
        sys.exit(1)

    # Initialize keyboard detector
    try:
        keyboard = PianoKeyboard()
        detector = KeyboardDetector(template_path)
    except Exception as e:
        print(f"Failed to initialize keyboard detector: {e}")
        sys.exit(1)

    vidcap = cv2.VideoCapture(input_video)
    success, image = vidcap.read()
    count = 0
    last_mod = 0

    if not success:
        print(f"Could not open video: {input_video}")
        sys.exit(1)

    fps = vidcap.get(cv2.CAP_PROP_FPS)
    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    print(f'Processing entire video: {duration:.1f} seconds at {fps:.1f} fps ({total_frames} frames)...')

    last_pressed = []
    keyboard_detected = False

    while success:
        # Detect keyboard on first frame
        if count == 0:
            print("Detecting keyboard layout...")
            if not detector.detect(image, keyboard):
                print("Failed to detect keyboard. Please check:")
                print("1. Template file exists and is valid")
                print("2. Video contains a visible piano")
                print("3. Piano is clearly visible in the first frame")
                sys.exit(1)

            # Verify detection quality
            try:
                detector.verify_middle_c(keyboard)
                print("Keyboard detection successful!")
            except ValueError as e:
                print(f"Keyboard detection verification failed: {e}")
                print("Detection may be inaccurate. Consider using a different template.")
                sys.exit(1)

            # Store default brightness values for comparison
            for key in keyboard:
                key.default_brightness = key.brightness

            # Create visualization of detected keys
            vis_image = image.copy()
            if detector.piano_boundary:
                x, y, w, h = detector.piano_boundary
                cv2.rectangle(vis_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Draw key positions
            for key in keyboard:
                if key.x is not None and key.y is not None:
                    color = (255, 255, 255) if key.type == 'W' else (0, 0, 0)
                    cv2.circle(vis_image, (key.x, key.y), 3, color, -1)
                    cv2.circle(vis_image, (key.x, key.y), 5, (0, 255, 0), 1)

            cv2.imwrite("keyboard_detection.jpg", vis_image)
            print("Keyboard detection visualization saved as 'keyboard_detection.jpg'")

            keyboard_detected = True
            last_pressed = [0] * len(keyboard)

        if keyboard_detected:
            # Sample current brightness at all key positions
            sample_current_brightness(image, keyboard)

            # Detect pressed keys
            pressed = get_pressed_keys(keyboard, threshold)

            # Generate MIDI events for key state changes
            for i, (current_pressed, last_state) in enumerate(zip(pressed, last_pressed)):
                if current_pressed != last_state:
                    key = keyboard[i]

                    # Calculate MIDI note number (A0 = 21, so key index + 21)
                    midi_note = i + 21

                    if last_mod == 0 and count > fps:
                        last_mod = count - fps

                    if current_pressed == 1:
                        # Note on
                        track.append(Message('note_on', note=midi_note, velocity=64,
                                           time=int((count - last_mod) * (480 / fps))))
                        last_mod = count
                        # Debug: print pressed key
                        # print(f"Key pressed: {key.name} (MIDI {midi_note})")
                    else:
                        # Note off
                        track.append(Message('note_off', note=midi_note, velocity=127,
                                           time=int((count - last_mod) * (480 / fps))))
                        last_mod = count

            # Show progress
            progress_percent = (count / total_frames) * 100
            print(f"Processing frame {count}/{total_frames} ({progress_percent:.1f}%)...", end="\r")
            last_pressed = pressed

        success, image = vidcap.read()
        count += 1

    vidcap.release()
    mid.save(output)
    print(f"\nConversion complete! Saved as {output}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Convert piano videos to MIDI files by analyzing key presses",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    ap.add_argument('video', help='YouTube URL or local video file (.mp4)')
    ap.add_argument('-o', '--output', type=str, default='out.mid',
                    help='Output MIDI file name (default: out.mid)')
    ap.add_argument('-t', '--threshold', type=int, default=30,
                    help='Activation threshold for key press detection (default: 30)')
    ap.add_argument('--template', type=str, default='data/template/piano-88-keys-1_0.png',
                    help='Path to piano template file for detection')

    args = ap.parse_args()

    convert(args.video, args.output, args.threshold, args.template)

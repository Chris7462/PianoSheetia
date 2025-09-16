import sys
import argparse
import cv2
import numpy as np
from mido import Message, MidiFile, MidiTrack, MetaMessage
import yt_dlp
import os

# Constants
MIN_KEY_WIDTH = 3

def label_keys(keyboard, default_values, white_threshold, black_threshold):
    """Find and label the middle C key position"""
    cIndex = 0
    cs = []

    for i in range(len(default_values)-6):
        if(default_values[i] > white_threshold and
           default_values[i+1] > white_threshold and
           default_values[i+2] < black_threshold and
           default_values[i+3] > white_threshold and
           default_values[i+4] < black_threshold and
           default_values[i+5] > white_threshold and
           default_values[i+6] > white_threshold):
            cs.append(i+1)

    if len(cs) == 0:
        print("Did not detect a valid keyboard at the specified start, check your start time and keyboard height")
        sys.exit(2)

    middle_c = cs[int((len(cs))/2)]
    print("Recognized key", middle_c, "as middle C.")
    return middle_c

def get_pressed_keys(keys, default_values, activation_threshold):
    """Determine which keys are currently pressed based on brightness changes"""
    pressed = []
    for i in range(len(keys)):
        if abs(keys[i] - default_values[i]) > activation_threshold:
            pressed.append(1)
        else:
            pressed.append(0)
    return pressed

def extract_key_positions(keyboard):
    """Extract key positions and default brightness values from keyboard image"""
    key_positions = []
    default_values = []

    in_white_key = False
    in_black_key = False
    key_start = 0
    max_brightness = max(keyboard)
    min_brightness = min(keyboard)
    white_threshold = min_brightness + (max_brightness - min_brightness) * 0.6
    black_threshold = min_brightness + (max_brightness - min_brightness) * 0.4

    for i in range(len(keyboard)):
        b = keyboard[i]
        if b > white_threshold:
            if not in_white_key and not in_black_key:
                in_white_key = True
                key_start = i
        else:
            if in_white_key:
                in_white_key = False
                if i - key_start > MIN_KEY_WIDTH:
                    key_positions.append(int((key_start + i) / 2))
                    default_values.append(keyboard[int((key_start + i) / 2)])

        if b < black_threshold:
            if not in_black_key and not in_white_key:
                in_black_key = True
                key_start = i
        else:
            if in_black_key:
                in_black_key = False
                if (i - key_start) > MIN_KEY_WIDTH:
                    key_positions.append(int((key_start + i) / 2))
                    default_values.append(keyboard[int((key_start + i) / 2)])

    print("Detected", len(key_positions), "keys.")
    return key_positions, default_values, white_threshold, black_threshold

def download_video_with_ytdlp(url, output_dir='videos/'):
    """Download video using yt-dlp instead of pytube"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Configure yt-dlp options
    ydl_opts = {
        'format': 'mp4[height<=720]/best[height<=720]/best',  # Prefer 720p MP4 or best available
        'outtmpl': os.path.join(output_dir, '%(title)s.%(ext)s'),
        'restrictfilenames': True,  # Remove special characters from filename
        'noplaylist': True,  # Only download single video, not playlist
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            # Get video info first
            info = ydl.extract_info(url, download=False)
            title = info.get('title', 'video')

            # Clean filename
            safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).rstrip()
            filename = os.path.join(output_dir, f"{safe_title}.mp4")

            print(f"Downloading: {title}")

            # Download the video
            ydl.download([url])

            # Find the downloaded file (yt-dlp might modify the filename)
            for file in os.listdir(output_dir):
                if file.endswith('.mp4') and safe_title.replace(' ', '_') in file.replace(' ', '_'):
                    return os.path.join(output_dir, file)

            # Fallback: return the expected filename
            return filename

        except Exception as e:
            print(f"Error downloading video: {e}")
            return None

def convert(video, is_url, output="out.mid", start=0, end=-1, keyboard_height=0.85, threshold=30):
    """Main conversion function"""
    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)

    if is_url:
        print("Downloading video...")
        input_video = download_video_with_ytdlp(video)
        if input_video is None:
            print("Failed to download video")
            sys.exit(1)
    else:
        input_video = video

    vidcap = cv2.VideoCapture(input_video)
    success, image = vidcap.read()
    count = 0
    last_mod = 0

    if not success:
        print(f"Could not open video: {input_video}")
        sys.exit(1)

    frame_height, frame_width, _ = image.shape
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    print("Processing video at %dp@%.1f fps..." % (frame_height, fps))

    keyboard_height_px = int(frame_height * keyboard_height)
    start_frame = int(start * fps)
    end_frame = int(end * fps) if end > 0 else -1

    last_pressed = []
    key_positions = []
    default_values = []
    middle_c = 0

    while success:
        ia = np.asarray(image)
        kb = []

        for x in range(len(ia[0])):
            kb.append(np.mean(ia[keyboard_height_px][x]))

        if count == start_frame:
            key_positions, default_values, white_threshold, black_threshold = extract_key_positions(kb)

            cv2.line(image, (0, keyboard_height_px), (frame_width, keyboard_height_px), (0, 255, 0), 2)
            for i in range(len(key_positions)):
                cv2.circle(image, (key_positions[i], keyboard_height_px), 7,
                          (255, 255, 255) if default_values[i] < white_threshold else (0, 0, 0), -1)
                cv2.circle(image, (key_positions[i], keyboard_height_px), 5,
                          (255, 255, 255) if default_values[i] > white_threshold else (0, 0, 0), -1)

            cv2.imwrite("start_frame.jpg", image)
            middle_c = label_keys(kb, default_values, white_threshold, black_threshold)

            last_pressed = [0] * len(key_positions)

        if count >= start_frame and len(key_positions) > 0:
            keys = []

            for i in range(len(key_positions)):
                keys.append(kb[key_positions[i]])

            pressed = get_pressed_keys(keys, default_values, threshold)

            for i in range(len(pressed)):
                if not pressed[i] == last_pressed[i]:
                    if last_mod == 0 and count > fps:
                        last_mod = count - fps
                    if pressed[i] == 1:
                        track.append(Message('note_on', note=60 - middle_c + i, velocity=64,
                                           time=int((count - last_mod) * (480 / fps))))
                        last_mod = count
                    if pressed[i] == 0:
                        track.append(Message('note_off', note=60 - middle_c + i, velocity=127,
                                           time=int((count - last_mod) * (480 / fps))))
                        last_mod = count
            print(f"Processing frame {count}...", end="\r")
            last_pressed = pressed

        success, image = vidcap.read()
        count += 1

        if end_frame > 0 and count > end_frame:
            break

    vidcap.release()
    mid.save(output)
    print(f"Saved as {output}!               ")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Convert piano videos to MIDI files by analyzing key presses",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s "https://youtube.com/watch?v=xyz" -o song.mid
  %(prog)s video.mp4 -s 30 -e 120 -t 40
  %(prog)s piano_video.mp4 -k 0.8 --threshold 25
        """
    )

    ap.add_argument('video',
                    help='YouTube URL or local MP4 file path')

    ap.add_argument('-o', '--output',
                    type=str,
                    default='out.mid',
                    help='Output MIDI file name (default: out.mid)')

    ap.add_argument('-s', '--start',
                    type=float,
                    default=0,
                    help='Start time in seconds (default: 0)')

    ap.add_argument('-e', '--end',
                    type=float,
                    default=-1,
                    help='End time in seconds (default: -1 for entire video)')

    ap.add_argument('-k', '--keyboard-height',
                    type=float,
                    default=0.85,
                    help='Proportional keyboard height from top (default: 0.85)')

    ap.add_argument('-t', '--threshold',
                    type=int,
                    default=30,
                    help='Activation threshold for key press detection (default: 30)')

    args = ap.parse_args()

    # Determine if input is URL or local file
    is_url = not args.video.endswith('.mp4')

    # Validate arguments
    if args.keyboard_height <= 0 or args.keyboard_height >= 1:
        print("Error: Keyboard height must be between 0 and 1")
        sys.exit(1)

    if args.threshold <= 0:
        print("Error: Threshold must be positive")
        sys.exit(1)

    if args.start < 0:
        print("Error: Start time cannot be negative")
        sys.exit(1)

    if args.end != -1 and args.end <= args.start:
        print("Error: End time must be greater than start time")
        sys.exit(1)

    convert(args.video, is_url, args.output, args.start, args.end,
            args.keyboard_height, args.threshold)

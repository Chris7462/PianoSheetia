import sys
import os
import yt_dlp


class VideoProcessor:
    """Handles video file operations and frame processing"""

    def __init__(self, output_dir: str = 'data/videos/'):
        self.output_dir = output_dir

    def video_acquisition(self, url, output_dir=None):
        """Download video using yt-dlp and return the actual saved filename"""
        if output_dir is None:
            output_dir = self.output_dir

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
                # Extract info and download in one step
                info = ydl.extract_info(url, download=True)

                # Get actual filename (yt-dlp ensures this is accurate after postprocessing)
                if 'requested_downloads' in info:
                    actual_filename = info['requested_downloads'][0]['_filename']
                else:
                    actual_filename = ydl.prepare_filename(info)

                return actual_filename

            except Exception as e:
                print(f'Error downloading video: {e}')
                return None

    def get_video_file(self, video_path: str) -> str:
        """Get video file, downloading if it's a URL"""
        is_url = not video_path.endswith(".mp4")

        if is_url:
            print("Downloading video...")
            input_video = self.video_acquisition(video_path)
            if input_video is None:
                print("Failed to download video")
                sys.exit(1)
            return input_video
        else:
            return video_path

#   def create_visualization(self, image, keyboard_processor: KeyboardProcessor,
#                            keyboard_height_px: int, left_bound: int, right_bound: int):
#       """Create start frame visualization"""
#       # Draw visualization on full frame
#       cv2.line(image, (left_bound, keyboard_height_px), (right_bound, keyboard_height_px), (0, 255, 0), 2)

#       # Draw keyboard region rectangle
#       cv2.rectangle(image, (left_bound, keyboard_height_px - 20), (right_bound, keyboard_height_px + 20), (255, 0, 0), 1)

#       # Adjust key positions to full frame coordinates for visualization
#       for i in range(len(keyboard_processor.key_positions)):
#           adjusted_x = left_bound + keyboard_processor.key_positions[i]
#           cv2.circle(image, (adjusted_x, keyboard_height_px), 7,
#                      (255,255,255) if keyboard_processor.default_values[i] < white_threshold else (0,0,0), -1)
#           cv2.circle(image, (adjusted_x, keyboard_height_px), 5,
#                      (255,255,255) if keyboard_processor.default_values[i] > white_threshold else (0,0,0), -1)

#       cv2.imwrite("start_frame.jpg", image)
#       print(f"Key detection visualization saved as: start_frame.jpg")

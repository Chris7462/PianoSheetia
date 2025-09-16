"""
video_downloader.py

Provides the VideoDownloader class for handling video downloads
using yt-dlp, including saving to a local directory and retrieving
file paths from either local files or URLs.
"""

import os
import yt_dlp


class VideoDownloader:
    """Handles video file operations and frame processing"""

    def __init__(self, output_dir: str = 'data/videos/'):
        self.output_dir = output_dir
        # Create output directory once during initialization
        os.makedirs(self.output_dir, exist_ok=True)

    def download_video(self, url: str) -> str:
        """Download video using yt-dlp and return the actual saved filename"""
        # Configure yt-dlp options
        ydl_opts = {
            'format': 'mp4[height<=720]/best[height<=720]/best', # Prefer 720p MP4 or best available
            'outtmpl': os.path.join(self.output_dir, '%(title)s.%(ext)s'),
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

            except yt_dlp.DownloadError as e:
                print(f'Download error: {e}')
                return None
            except (OSError, IOError) as e:
                print(f'File system error: {e}')
                return None
            except Exception as e:  # pylint: disable=broad-exception-caught
                print(f'Unexpected error downloading video: {e}')
                return None

    def get_video_file(self, path_or_url: str) -> str:
        """Get video file, downloading if it's a URL"""
        # More reliable URL detection
        if path_or_url.startswith(('http://', 'https://')):
            print("Downloading video...")
            downloaded_file = self.download_video(path_or_url)
            if downloaded_file is None:
                print("Failed to download video")
            return downloaded_file

        return path_or_url

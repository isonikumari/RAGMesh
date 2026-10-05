import os
import shutil
import sys


def ensure_ffmpeg_on_path() -> None:
    ffmpeg_dirs = [
        r"C:\Users\misha\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-9.0-full_build\bin",
        r"C:\Program Files\ffmpeg\bin",
        r"C:\Program Files (x86)\ffmpeg\bin",
    ]
    path_entries = os.environ.get("PATH", "").split(os.pathsep)
    for ffmpeg_dir in ffmpeg_dirs:
        if os.path.isdir(ffmpeg_dir) and ffmpeg_dir not in path_entries:
            os.environ["PATH"] = os.environ.get("PATH", "") + os.pathsep + ffmpeg_dir

    if shutil.which("ffmpeg") is None and shutil.which("avconv") is None:
        raise RuntimeError(
            "FFmpeg is not installed or not on PATH. Install FFmpeg and restart the terminal."
        )


ensure_ffmpeg_on_path()

import yt_dlp
from pydub import AudioSegment

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Try to use a location with more space (temp directory or local appdata)
# Falls back to project directory if temp not available
PREFERRED_DOWNLOAD_DIRS = [
    os.path.join(os.environ.get('TEMP', ''), 'ai_video_downloads'),
    os.path.join(os.path.expanduser('~'), '.ai_video_assistant', 'downloads'),
    os.path.join(PROJECT_ROOT, 'downloades')
]

DOWNLOAD_DIR = None
for dir_path in PREFERRED_DOWNLOAD_DIRS:
    try:
        if dir_path and os.path.dirname(dir_path):
            os.makedirs(dir_path, exist_ok=True)
            # Test if we can write to this directory
            test_file = os.path.join(dir_path, '.test_write')
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)
            DOWNLOAD_DIR = dir_path
            print(f"Using download directory: {DOWNLOAD_DIR}")
            break
    except (OSError, Exception):
        continue

if DOWNLOAD_DIR is None:
    # Fallback to project directory
    DOWNLOAD_DIR = os.path.join(PROJECT_ROOT, 'downloades')
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    print(f"Fallback download directory: {DOWNLOAD_DIR}")


def download_youtube_audio(url: str) -> str:
    output_path = os.path.join(DOWNLOAD_DIR, "%(title)s.%(ext)s")
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_path,
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "extractor_args": {
            "youtube": {
                "player_client": ["web", "android", "tv_embedded"],
            }
        },
        "http_headers": {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
        },
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "wav",
            "preferredquality": "192",
        }],
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
        filename = os.path.splitext(ydl.prepare_filename(info))[0] + ".wav"
        if not os.path.isfile(filename):
            raise RuntimeError(f"Downloaded audio was not found at: {filename}")
        return filename
    except yt_dlp.utils.DownloadError as exc:
        raise RuntimeError(
            "YouTube download failed. The video may be private, age-restricted, region-blocked, or temporarily blocked by the source. "
            f"Original error: {exc}"
        ) from exc


def convert_to_wav(input_path: str) -> str:
    """Convert an audio/video file to mono, 16 kHz WAV."""
    output_path = os.path.splitext(input_path)[0] + "_converted.wav"
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_channels(1).set_frame_rate(16000)
    audio.export(output_path, format="wav")
    return output_path


def chunk_audio(wav_path: str, chunk_minutes: int = 5) -> list[str]:
    """Split a WAV file into chunks and return the generated file paths.
    
    Args:
        wav_path: Path to the WAV file to chunk
        chunk_minutes: Size of each chunk in minutes (default: 5 to reduce memory)
    """
    if chunk_minutes <= 0:
        raise ValueError("chunk_minutes must be greater than zero")

    audio = AudioSegment.from_wav(wav_path)
    chunk_ms = chunk_minutes * 60 * 1000
    base_path = os.path.splitext(wav_path)[0]
    chunks = []

    for index, start in enumerate(range(0, len(audio), chunk_ms)):
        chunk = audio[start:start + chunk_ms]
        chunk_path = f"{base_path}_chunk_{index}.wav"
        try:
            chunk.export(chunk_path, format="wav")
            chunks.append(chunk_path)
        except OSError as e:
            if "No space left on device" in str(e):
                print(f"Warning: Disk space low. Exported {index} chunks before running out of space.")
                raise RuntimeError(
                    f"Insufficient disk space. Only {index} chunks could be created. "
                    f"Free up space and try again, or use a service with more storage."
                ) from e
            raise

    return chunks


if __name__ == "__main__":
    video_url = sys.argv[1] if len(sys.argv) > 1 else "https://www.youtube.com/watch?v=OYvlznJ4IZQ"
    data = download_youtube_audio(video_url)
    converted_path = convert_to_wav(data)
    # print(converted_path)
    # print(chunk_audio(converted_path))

def cleanup_old_files(directory: str = None, pattern: str = "_chunk_") -> int:
    """Clean up old audio chunk files to free disk space.
    
    Args:
        directory: Directory to clean (default: DOWNLOAD_DIR)
        pattern: File pattern to match for deletion (default: "_chunk_")
    
    Returns:
        Number of files deleted
    """
    if directory is None:
        directory = DOWNLOAD_DIR
    
    if not os.path.isdir(directory):
        return 0
    
    deleted_count = 0
    for file in os.listdir(directory):
        if pattern in file and file.endswith('.wav'):
            try:
                file_path = os.path.join(directory, file)
                os.remove(file_path)
                deleted_count += 1
            except Exception as e:
                print(f"Warning: Could not delete {file}: {e}")
    
    return deleted_count


def process_input(source: str) -> list[str]:
    """Download or convert an input file, then split it into audio chunks."""
    # Clean up old chunks before starting
    cleaned = cleanup_old_files()
    if cleaned > 0:
        print(f"Cleaned up {cleaned} old audio files to free space")
    
    if source.startswith(("https://", "http://")):
        print("Detected YouTube URL. Downloading audio...")
        wav_path = download_youtube_audio(source)
    else:
        print("Detected local file. Converting to WAV...")
        wav_path = convert_to_wav(source)

    print("Chunking audio...")
    try:
        chunks = chunk_audio(wav_path)
        print(f"Audio ready - {len(chunks)} chunk(s) created.")
        return chunks
    except RuntimeError as e:
        print(f"Error: {e}")
        raise
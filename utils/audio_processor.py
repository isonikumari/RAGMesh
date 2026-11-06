import os
import shutil


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

from pydub import AudioSegment


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
    """Convert a local audio/video file and split it into audio chunks."""
    if source.startswith(("https://", "http://")):
        raise ValueError("URL inputs are not supported. Upload a local audio or video file.")

    cleaned = cleanup_old_files()
    if cleaned > 0:
        print(f"Cleaned up {cleaned} old audio files to free space")

    print("Converting local file to WAV...")
    wav_path = convert_to_wav(source)

    print("Chunking audio...")
    try:
        chunks = chunk_audio(wav_path)
        print(f"Audio ready - {len(chunks)} chunk(s) created.")
        return chunks
    except RuntimeError as e:
        print(f"Error: {e}")
        raise
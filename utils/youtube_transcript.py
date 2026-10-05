import re
from urllib.parse import parse_qs, urlparse


_VIDEO_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{11}$")
_YOUTUBE_HOSTS = {
    "youtube.com",
    "www.youtube.com",
    "m.youtube.com",
    "music.youtube.com",
    "youtube-nocookie.com",
    "www.youtube-nocookie.com",
}


def extract_youtube_video_id(url: str) -> str:
    parsed_url = urlparse(url.strip())
    hostname = (parsed_url.hostname or "").lower()

    if hostname == "youtu.be":
        video_id = parsed_url.path.strip("/").split("/", maxsplit=1)[0]
    elif hostname in _YOUTUBE_HOSTS:
        path_parts = [part for part in parsed_url.path.split("/") if part]
        if parsed_url.path.rstrip("/") == "/watch":
            video_id = parse_qs(parsed_url.query).get("v", [""])[0]
        elif len(path_parts) >= 2 and path_parts[0] in {"embed", "live", "shorts"}:
            video_id = path_parts[1]
        else:
            video_id = ""
    else:
        video_id = ""

    if not _VIDEO_ID_PATTERN.fullmatch(video_id):
        raise ValueError("Enter a valid YouTube video, Shorts, or live-stream URL.")
    return video_id


def fetch_youtube_transcript(url: str, translate_to_english: bool = False) -> str | None:
    video_id = extract_youtube_video_id(url)

    from youtube_transcript_api import YouTubeTranscriptApi
    from youtube_transcript_api._errors import (
        NoTranscriptFound,
        YouTubeTranscriptApiException,
    )

    try:
        transcript_list = YouTubeTranscriptApi().list(video_id)
        try:
            transcript = transcript_list.find_transcript(["en"])
        except NoTranscriptFound:
            transcript = next(iter(transcript_list), None)

        if transcript is None:
            return None

        if (
            translate_to_english
            and transcript.language_code != "en"
            and transcript.is_translatable
        ):
            transcript = transcript.translate("en")

        fetched_transcript = transcript.fetch()
    except YouTubeTranscriptApiException as exc:
        print(f"YouTube captions are unavailable; trying audio download instead: {exc}")
        return None

    text = " ".join(
        snippet.text.strip()
        for snippet in fetched_transcript
        if snippet.text.strip()
    )
    return re.sub(r"\s+", " ", text).strip() or None

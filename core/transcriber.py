import os
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def get_client():
    from google import genai

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is required for Gemini transcription.")

    return genai.Client(api_key=api_key)


def transcribe_chunk(chunk_path: str, translate: bool = False) -> str:
    from google.genai import types

    if translate:
        prompt = "Transcribe this audio and translate the speech into English. Return only the spoken words."
    else:
        prompt = "Transcribe this audio in its original language. Return only the spoken words."

    model = (
        os.getenv("GEMINI_TRANSCRIPTION_MODEL", "").strip()
        or os.getenv("GEMINI_MODEL", "gemini-3.6-flash").strip()
    )
    if not model:
        raise RuntimeError("Set GEMINI_TRANSCRIPTION_MODEL or GEMINI_MODEL to a Gemini model name.")

    response = get_client().models.generate_content(
        model=model,
        contents=[
            prompt,
            types.Part.from_bytes(
                data=Path(chunk_path).read_bytes(),
                mime_type="audio/wav",
            ),
        ],
    )
    transcript = response.text
    if not transcript or not transcript.strip():
        raise RuntimeError(f"Gemini returned an empty transcript for audio chunk: {chunk_path}")
    return transcript.strip()


def transcribe_all(chunks: list[str], translate: bool = False) -> str:
    transcripts = []
    for i, chunk in enumerate(chunks):
        print(f"Transcribing chunk {i + 1}")
        transcripts.append(transcribe_chunk(chunk, translate=translate))
        print("Transcription Completed")
    return " ".join(transcripts)

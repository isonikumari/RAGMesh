# RAGMesh

AI Meeting Assistant is a Python application that ingests a YouTube URL or local audio/video file, converts it into a transcript, summarizes the content, extracts key action items and decisions, and answers questions against the meeting context using a retrieval-augmented generation (RAG) pipeline.

The project combines Streamlit for the interface, FFmpeg and `yt-dlp` for media processing, Google Gemini for transcription and LLM tasks, and ChromaDB for vector-based retrieval.

## Features

- Upload local video/audio files or paste a YouTube link
- Try available YouTube captions first, then download and transcribe audio when captions are unavailable
- Convert media to WAV format and split long recordings into smaller chunks
- Transcribe chunks with Gemini and combine them into a full transcript
- Summarize the transcript into a clean, professional outcome
- Extract:
  - action items
  - key decisions
  - open questions
- Ask questions about the meeting content using a context-aware RAG pipeline
- Persist vector embeddings in local ChromaDB storage for retrieval

## Project overview

This app is designed for meetings, lectures, podcasts, interviews, and other spoken content where quick summaries and searchable context are useful.

The general workflow is:

1. Input a YouTube URL or local file
2. For YouTube, try available captions first; if unavailable, download/convert the media to WAV
3. Split downloaded audio into chunks to keep processing manageable
4. Transcribe each audio chunk with Gemini
5. Merge transcript chunks into one transcript
6. Generate a title and summary
7. Extract action items, decisions, and open questions
8. Build a vector store from the cleaned transcript
9. Answer user questions using relevant retrieved context

## Tech stack

- Python 3.12.10 (pinned for Render; Python 3.13+ removes the `audioop` module used by `pydub`)
- Streamlit
- LangChain
- ChromaDB
- Google Gemini API
- FFmpeg
- `yt-dlp`
- `pydub`
- `youtube-transcript-api`
- `python-dotenv`

## Repository structure

```text
RAGMesh/
├── app.py                  # Main Streamlit UI
├── main.py                 # Duplicate entry point / alternate runner
├── .python-version         # Python version used for deployment
├── .env                    # Local environment variables
├── .gitignore
├── requirements.txt        # Python dependencies
├── render.yaml             # Render deployment configuration
├── core/
│   ├── extractor.py        # Extract action items, questions, decisions
│   ├── mistral.py          # Gemini/LLM configuration and rate-limit handling
│   ├── rag_engine.py       # Retrieval-augmented generation pipeline
│   ├── summarize.py        # Transcript summary and title generation
│   ├── transcriber.py      # Gemini transcription wrapper
│   ├── vector_store.py     # ChromaDB vector helper utilities
│   └── __pycache__
├── utils/
│   ├── audio_processor.py  # Download, conversion, chunking, and cleanup logic
│   └── youtube_transcript.py # Caption retrieval and YouTube URL parsing
├── vector_db/              # ChromaDB persistence directory
└── README.MD
```

## Prerequisites

Before running the project, make sure you have:

- Python installed
- FFmpeg installed and available on your `PATH`
- A valid Google Gemini API key

### Installing FFmpeg

On Windows, the project expects FFmpeg to be present on the system PATH. A common setup is:

- Download FFmpeg from https://www.ffmpeg.org/download.html
- Add the `bin` folder to your PATH
- Restart the terminal after installation

Example Windows paths often look like:

```powershell
C:\ffmpeg\bin
```

## Environment variables

Create a `.env` file in the project root with the following values:

```env
GOOGLE_API_KEY="your_google_gemini_api_key"
GEMINI_MODEL=gemini-3.6-flash
GEMINI_TRANSCRIPTION_MODEL=gemini-3.6-flash
GEMINI_EMBEDDING_MODEL=gemini-embedding-001
```

Optional tuning values:

```env
GEMINI_MAX_RETRIES=5
GEMINI_REQUESTS_PER_SECOND=0.5
```

## Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/isonikumari/RAGMesh.git
cd RAGMesh
python -m venv .venv
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

On macOS/Linux:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

## Running the app locally

Start the Streamlit application:

```bash
streamlit run app.py
```

Then open the local browser URL shown in the terminal (usually `http://localhost:8501`).

## How the pipeline works

### 1. Media processing
The app accepts either a YouTube URL or a local file. For YouTube URLs, it first tries to retrieve captions with `youtube-transcript-api`; if captions cannot be accessed, it falls back to `yt-dlp` audio download and Gemini transcription. Local media is converted into WAV format and split into smaller chunks for manageable transcription.

### 2. Transcription
When captions are available, they are used directly. Otherwise, each audio chunk is sent to Gemini using a transcription prompt. The app can optionally translate non-English content into English before summarization and analysis.

### 3. Analysis and summarization
The combined transcript is passed through prompts that:

- generate a title
- write a concise summary
- extract action items
- extract key decisions
- capture follow-up/open questions

### 4. RAG-powered Q&A
The cleaned transcript is chunked, embedded, and stored in ChromaDB. User questions are answered by retrieving the most relevant transcript segments and prompting Gemini with the retrieved context.

## Notes on behavior

- The app is optimized for meeting-style transcripts and spoken content.
- The RAG layer filters clearly irrelevant personal chatter or social remarks so that answers stay focused on business context.
- Gemini rate limits may occur under heavy usage; the app includes retry logic and a rate-limit check.
- If YouTube content is private, region-blocked, or unsupported, download may fail.
- YouTube may block requests from cloud-hosted IP addresses. Caption retrieval and audio download are best-effort on Render; when both are blocked, upload the media file instead.

## Deployment

The project includes a Render configuration in `render.yaml`.

```yaml
services:
  - type: web
    name: ragify
    runtime: python
    buildCommand: apt-get update && apt-get install -y ffmpeg && pip install -r requirements.txt
    startCommand: streamlit run app.py --server.address=0.0.0.0 --server.port=$PORT
```

This makes it suitable for deployment on Render or similar Python hosting platforms with FFmpeg available.

## Example usage

1. Open the app in the browser
2. Paste a YouTube URL or upload a local video/audio file
3. Select a language preference
4. Press `Start Analysis`
5. Review the generated summary, action items, decisions, and Q&A output

## License

This project does not include a specific license file. Use it as a learning project or adapt it to your own workflow unless you have explicit permission to redistribute it under a different license.

## Troubleshooting

### FFmpeg not found
Make sure FFmpeg is installed and available in your `PATH`.

### Missing `GOOGLE_API_KEY`
Set `GOOGLE_API_KEY` in your environment or `.env` before launching the app.

### Transcription fails or returns empty output
Check that the input file is valid, the model name is correct, and your Google API quota/billing is active.

### Rate-limited requests
If Gemini is rate-limiting you, wait a bit and retry. The app includes a rate-limit detection path and a warning message for that condition.

## Future enhancements

Potential next improvements include:

- PDF/Markdown export of summaries and meeting notes
- Better speaker identification and diarization
- Multi-meeting comparison and search
- Dashboard for historical transcripts and saved sessions
- Support for more audio/video sources and longer recordings

## Summary

RAGMesh is a practical AI meeting assistant that turns spoken content into actionable knowledge. It combines transcription, summarization, extraction, and retrieval to make video and audio data searchable and easy to understand.

# RAGMesh

RAGMesh is a Python application for analyzing uploaded documents and local audio/video files. It extracts text or transcribes speech, summarizes the content, identifies key action items and decisions, and answers questions using a retrieval-augmented generation (RAG) pipeline.

The project combines Streamlit for the interface, FFmpeg for local media processing, Google Gemini for transcription and LLM tasks, and ChromaDB for vector-based retrieval.

## Features

- Upload a PDF or local video/audio file
- Extract selectable text from PDFs for summaries and questions
- Convert local media to WAV format and split long recordings into smaller chunks
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

1. Upload a PDF, audio file, or video file
2. For PDFs, extract selectable text directly; scanned image-only PDFs are not supported
3. Convert local audio/video to WAV and split audio into chunks
4. Transcribe each audio chunk with Gemini
5. Merge transcript chunks or extracted PDF text into one source document
6. Generate a title and summary
7. Extract action items, decisions, and open questions
8. Build a vector store from the cleaned content
9. Answer user questions using relevant retrieved context

## Tech stack

- Python 3.12.10 (pinned for Render; Python 3.13+ removes the `audioop` module used by `pydub`)
- Streamlit
- LangChain
- ChromaDB
- Google Gemini API
- FFmpeg
- `pydub`
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
│   └── audio_processor.py  # Local media conversion, chunking, and cleanup logic
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
PDF uploads are read as selectable text. Local audio and video files are converted to WAV and split into smaller chunks for transcription with Gemini.

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

### Evaluating retrieval
`evals/eval_retriever.py` evaluates the Chroma retriever with DeepEval's contextual precision and recall metrics. The included transcript and goldens are a small demonstration dataset; replace them with a representative transcript and manually verified questions and answers for your own evaluation.

From the project root, run the retrieval evaluation:

```powershell
.\.venv\Scripts\python.exe -m evals.eval_retriever
```

Both Gemini embeddings and the DeepEval judge use `GOOGLE_API_KEY` from `.env`. The judge defaults to `GEMINI_MODEL` (or `gemini-2.0-flash` if unset). The retriever evaluation uses `data/retriever_eval_transcript.txt` and `golden/retriever_goldens.json`.

### Evaluating generation
Run the generator evaluation with:

```powershell
.\.venv\Scripts\python.exe -m evals.eval_generator
```

It gives the generator fixed, known-good contexts from `golden/generator_goldens.json` and evaluates its answers with DeepEval's faithfulness and answer-relevancy metrics. This isolates answer generation from retrieval. It uses `GOOGLE_API_KEY` for both the app's Gemini generator and DeepEval judge, and waits between cases to reduce rate-limit errors.

### Evaluating the full RAG pipeline
Run the end-to-end retriever-plus-generator evaluation with:

```powershell
.\.venv\Scripts\python.exe -m evals.eval_rag_pipeline
```

It retrieves live Chroma context for each question, generates an answer using that context, and scores contextual relevancy, faithfulness, and answer relevancy. Unlike the generator-only evaluation, this tests retrieval and generation together.

### Additional evaluations
All evaluation modules live under `evals/` and run from the project root:

```powershell
.\.venv\Scripts\python.exe -m evals.eval_toxicity
.\.venv\Scripts\python.exe -m evals.eval_scope
.\.venv\Scripts\python.exe -m evals.eval_latency
.\.venv\Scripts\python.exe -m evals.eval_reliability
```

Toxicity and scope use DeepEval judges and therefore make Gemini API calls. Latency and reliability exercise the live retriever/generator without a judge; latency reports end-to-end, retrieval, and generation timing. The app does not stream answers, so it cannot measure TTFT. Reliability uses a single retry for non-rate-limit failures; rate-limit errors are not retried because another immediate request would not resolve quota exhaustion. These scripts use small demo question sets; expand them and choose suitable latency/reliability sample counts before treating results as representative.

### Evaluating cost
Run the token and generation-cost estimate with:

```powershell
.\.venv\Scripts\python.exe -m evals.eval_cost
```

This uses the real retriever context and the same prompt/model as application Q&A, then reads token usage from the Gemini response. It makes live generation calls (one per `golden/application_goldens.json` case) and waits between them. The estimate excludes embedding/retrieval, evaluation judges, and other provider charges. Prices are intentionally not hardcoded: set `GEMINI_INPUT_USD_PER_1M` and `GEMINI_OUTPUT_USD_PER_1M` in `.env` to current model rates to calculate a cost. Optionally set `GEMINI_CACHED_INPUT_USD_PER_1M`; otherwise cached input tokens are conservatively charged at the regular input rate. Without input/output prices the script reports tokens only.

### Evaluating application quality
Run correctness, completeness, and style checks against live RAG answers:

```powershell
.\.venv\Scripts\python.exe -m evals.eval_application
```

The demo cases are in `golden/application_goldens.json`. This calls the retriever and generator for each case and then uses a Gemini judge, so it requires `GOOGLE_API_KEY`, consumes API quota, and should not be treated as an offline/no-network test.

### Evaluating leakage
Run prompt-instruction, source-content, and PII leakage checks with:

```powershell
.\.venv\Scripts\python.exe -m evals.eval_leakage
```

The demo adversarial questions are in `golden/leakage_goldens.json`. Answers are generated by the live RAG pipeline; DeepEval's Gemini judge evaluates prompt/content handling and its PII metric checks generated answers. This also makes live API calls and requires `GOOGLE_API_KEY`. These small examples are demonstrations, not a security guarantee.

## Notes on behavior

- The app is optimized for meeting-style transcripts and spoken content.
- The RAG layer filters clearly irrelevant personal chatter or social remarks so that answers stay focused on business context.
- Gemini rate limits may occur under heavy usage; the app includes retry logic and a rate-limit check.

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
2. Upload a PDF, or a local video/audio file
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

### PDF text extraction fails
PDF uploads must contain selectable text. Image-only scanned PDFs are not supported because OCR is not enabled; use a text-searchable PDF instead.

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

RAGMesh is a practical assistant that turns uploaded documents and spoken content into actionable knowledge. It combines document extraction, transcription, summarization, and retrieval to make the source material searchable and easy to understand.

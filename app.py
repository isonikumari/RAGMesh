import os
import tempfile
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=True)

import streamlit as st

# ============================================================
# Cache the RAG chain as a resource
# ============================================================
@st.cache_resource
def load_rag_chain(transcript: str):
    from core.rag_engine import build_rag_chain

    return build_rag_chain(transcript)


# ============================================================
# Run the AI processing pipeline
# ============================================================
@st.cache_data(show_spinner=True)
def run_pipeline(source: str, language: str = "english") -> dict:
    from core.extractor import (
        extract_action_items,
        extract_key_decisions,
        extract_question,
    )
    from core.summarize import generate_title, summarize
    from core.transcriber import transcribe_all
    from utils.audio_processor import process_input

    st.info("Starting content analysis...")

    transcript = None
    is_pdf = Path(source).suffix.lower() == ".pdf"
    progress_bar = st.progress(0, text="Preparing PDF analysis...") if is_pdf else None

    if is_pdf:
        from utils.pdf_processor import extract_pdf_text

        st.info("Extracting text from the PDF...")
        def update_pdf_progress(page_number: int, total_pages: int) -> None:
            if progress_bar is not None:
                progress_bar.progress(
                    0.1 * page_number / total_pages,
                    text=f"Extracting PDF text: page {page_number} of {total_pages}",
                )

        transcript = extract_pdf_text(source, progress_callback=update_pdf_progress)
        st.success("PDF text extracted.")
    if not transcript:
        chunks = process_input(source)
        transcript = transcribe_all(chunks, translate=(language != "english"))

    if not is_pdf:
        st.success("Transcription completed!")

    if progress_bar is not None:
        progress_bar.progress(0.12, text="Generating a title...")
    title = generate_title(transcript)
    if progress_bar is not None:
        progress_bar.progress(0.15, text="Summarizing PDF content...")

    def update_summary_progress(chunk_number: int, total_chunks: int) -> None:
        if progress_bar is not None:
            progress_bar.progress(
                0.15 + 0.6 * chunk_number / total_chunks,
                text=f"Summarizing PDF content: section {chunk_number} of {total_chunks}",
            )

    summary = summarize(
        transcript,
        progress_callback=update_summary_progress if is_pdf else None,
    )
    if progress_bar is not None:
        progress_bar.progress(0.76, text="Extracting action items...")
    action_items = extract_action_items(transcript)
    if progress_bar is not None:
        progress_bar.progress(0.83, text="Extracting key decisions...")
    decisions = extract_key_decisions(transcript)
    if progress_bar is not None:
        progress_bar.progress(0.90, text="Finding open questions...")
    questions = extract_question(transcript)
    if progress_bar is not None:
        progress_bar.progress(1.0, text="PDF analysis complete.")

    return {
        "title": title,
        "transcript": transcript,
        "summary": summary,
        "action_items": action_items,
        "key_decisions": decisions,
        "open_questions": questions,
    }


# ============================================================
# Helper to handle uploaded local files
# ============================================================
def save_uploaded_file(uploaded_file) -> str | None:
    if uploaded_file is None:
        return None

    suffix = Path(uploaded_file.name).suffix.lower() or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(uploaded_file.getvalue())
        return temp_file.name


# ============================================================
# Streamlit UI
# ============================================================
st.set_page_config(
    page_title="AI Meeting Assistant",
    page_icon="🎥",
    layout="wide",
)

st.markdown(
    """
    <style>
        .stApp {
            background-image: linear-gradient(rgba(255,255,255,0.78), rgba(255,255,255,0.78)), url("https://img.freepik.com/premium-photo/abstract-background-with-shining-blue-neural-network-system-wallpaper-connected-lines-glowing-dots-particles-close-up-view-horizontal-illustration-banner-design-generative-ai_9209-12822.jpg");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
            color: #0f172a;
        }

        .app-header {
            background: linear-gradient(135deg, #e0ecff 0%, #f1e8ff 100%);
            border: 1px solid rgba(148, 163, 184, 0.35);
            border-radius: 22px;
            padding: 1.5rem 1.5rem 1.25rem 1.5rem;
            box-shadow: 0 12px 28px rgba(37, 99, 235, 0.12);
            margin-bottom: 1rem;
        }

        .app-header h1 {
            margin: 0;
            color: #111827;
            font-size: 2.3rem;
            font-weight: 700;
        }

        .app-header p {
            margin: 0.5rem 0 0 0;
            color: #374151;
            font-size: 1rem;
        }

        .panel {
            background: rgba(255, 255, 255, 0.95);
            border: 1px solid #e2e8f0;
            border-radius: 18px;
            padding: 1rem 1.2rem;
            box-shadow: 0 8px 18px rgba(15, 23, 42, 0.04);
            margin-bottom: 1rem;
        }

        .panel h3 {
            margin-top: 0;
            margin-bottom: 0.55rem;
            color: #0f172a;
        }

        .panel p, .panel li {
            color: #334155;
            line-height: 1.7;
        }

        .stTextInput input,
        .stTextArea textarea,
        .stSelectbox div[role="combobox"],
        .stNumberInput input,
        .stChatInput textarea,
        .stChatInput input {
            color: #0b1120 !important;
            background: rgba(255, 255, 255, 0.96) !important;
            border-color: rgba(99, 102, 241, 0.55) !important;
        }

        .stTextInput input::placeholder,
        .stTextArea textarea::placeholder,
        .stChatInput textarea::placeholder,
        .stChatInput input::placeholder {
            color: #4b5563 !important;
            opacity: 1;
        }

        .stTextInput label,
        .stTextArea label,
        .stSelectbox label,
        .stNumberInput label,
        .stRadio > div,
        .stRadio label,
        .stMarkdown {
            color: #111827 !important;
            font-weight: 600;
        }

        div[data-testid="stChatMessage"] {
            color: #111827 !important;
        }

        div[data-testid="stChatMessage"] p,
        div[data-testid="stChatMessage"] div {
            color: #111827 !important;
        }

        div[data-testid="stSidebar"] > div {
            background: #f8fafc;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="app-header">
        <h1>🎥 AI Meeting Assistant</h1>
        <p>Analyze documents and local audio or video files, then ask questions about their content.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

uploaded_file = None

uploaded_file = st.file_uploader(
    "Upload a document, audio, or video file",
    type=["pdf", "mp4", "mp3", "wav", "m4a", "mov", "avi", "webm", "mpeg", "ogg"],
    help="Upload a PDF with selectable text or a local audio/video file to analyze.",
)

language = st.selectbox("Language", ["english", "hindi", "other"])
process_button = st.button("Start Analysis", use_container_width=True, type="primary")

with st.sidebar:
    st.title("⚙️ Settings")
    st.caption("Upload a document or local media file to get started.")


if process_button:
    final_source = None

    if uploaded_file is None:
        st.warning("Please upload a document, audio, or video file.")
        st.stop()
    final_source = save_uploaded_file(uploaded_file)

    with st.spinner("Processing your file and generating results..."):
        try:
            result = run_pipeline(final_source, language)
        except Exception as exc:
            from core.mistral import is_rate_limit_error

            if is_rate_limit_error(exc):
                st.error(
                    "Gemini is rate-limiting this request. Wait a minute and retry. "
                    "If it continues, check your Google Gemini API usage, billing, or API key plan."
                )
                st.stop()
            from utils.pdf_processor import PDFExtractionError

            if isinstance(exc, PDFExtractionError):
                st.error(str(exc))
                st.stop()
            raise

    st.session_state["pipeline_result"] = result
    st.session_state["rag_chain"] = load_rag_chain(result["transcript"])
    st.session_state["chat_history"] = []


if "pipeline_result" in st.session_state:
    result = st.session_state["pipeline_result"]

    col_title, col_summary = st.columns([1.1, 1.9])

    with col_title:
        st.markdown(
            f"""
            <div class="panel">
                <h3>📌 Video Title</h3>
                <p>{result['title']}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_summary:
        st.markdown(
            f"""
            <div class="panel">
                <h3>📝 Summary</h3>
                <p>{result['summary']}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("""
        <div class="panel">
            <h3>💬 Ask a question about the uploaded content</h3>
        </div>
    """, unsafe_allow_html=True)

    for message in st.session_state.get("chat_history", []):
        if message["role"] == "user":
            with st.chat_message("user"):
                st.write(message["content"])
        else:
            with st.chat_message("assistant"):
                st.write(message["content"])

    user_question = st.chat_input("Ask a question about this content")
    if user_question:
        from core.rag_engine import ask_question

        rag_chain = st.session_state.get("rag_chain") or load_rag_chain(result["transcript"])
        st.session_state["rag_chain"] = rag_chain

        with st.spinner("Thinking..."):
            answer = ask_question(rag_chain, user_question)

        st.session_state["chat_history"] = st.session_state.get("chat_history", [])
        st.session_state["chat_history"].append({"role": "user", "content": user_question})
        st.session_state["chat_history"].append({"role": "assistant", "content": answer})
        st.rerun()
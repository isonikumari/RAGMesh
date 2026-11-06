import os
from pathlib import Path

from dotenv import load_dotenv

from core.rag_engine import generate_answer, sanitize_meeting_context
from core.vector_store import ensure_vector_store, get_collection_name, get_retriever


ROOT_DIR = Path(__file__).resolve().parent.parent
TRANSCRIPT_PATH = ROOT_DIR / "data" / "retriever_eval_transcript.txt"
JUDGE_MODEL = os.getenv("DEEPEVAL_JUDGE_MODEL", os.getenv("GEMINI_MODEL", "gemini-2.0-flash"))


def load_eval_retriever(k: int = 10):
    load_dotenv(ROOT_DIR / ".env")
    if not os.getenv("GOOGLE_API_KEY"):
        raise RuntimeError("Set GOOGLE_API_KEY in .env to run the evaluation.")

    transcript = sanitize_meeting_context(TRANSCRIPT_PATH.read_text(encoding="utf-8"))
    store = ensure_vector_store(transcript, get_collection_name(transcript))
    return get_retriever(store, k=k)


def run_question(retriever, query: str):
    documents = retriever.invoke(query)
    if not documents:
        raise RuntimeError(f"No documents retrieved for: {query}")
    context = [document.page_content for document in documents]
    return generate_answer(query, context), context

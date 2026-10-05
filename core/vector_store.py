import hashlib
import os
from functools import lru_cache

from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

CHROMA_DIR = os.path.abspath("vector_db")
COLLECTION_NAME = "meeting_transcript"
EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")


@lru_cache(maxsize=2)
def get_embeddings():
  return GoogleGenerativeAIEmbeddings(
    model=EMBEDDING_MODEL,
    google_api_key=os.getenv("GOOGLE_API_KEY"),
  )


def get_collection_name(transcript: str | None = None) -> str:
  if transcript is None:
    return COLLECTION_NAME

  transcript_digest = hashlib.sha256(transcript.strip().encode("utf-8")).hexdigest()[:12]
  model_digest = hashlib.sha256(EMBEDDING_MODEL.encode("utf-8")).hexdigest()[:8]
  return f"{COLLECTION_NAME}_{model_digest}_{transcript_digest}"


def build_vector_store(transcript: str, collection_name: str | None = None) -> Chroma:
  print("Building vector store")
  collection_name = collection_name or get_collection_name(transcript)

  splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
  )
  chunks = splitter.split_text(transcript)

  docs = [Document(page_content=chunk, metadata={'chunk_index': i})
          for i, chunk in enumerate(chunks)]

  embeddings = get_embeddings()
  vector_store = Chroma.from_documents(
      documents=docs,
      embedding=embeddings,
      collection_name=collection_name,
      persist_directory=CHROMA_DIR
  )

  return vector_store


def load_vector_store(collection_name: str | None = None) -> Chroma:
  embeddings = get_embeddings()
  collection_name = collection_name or COLLECTION_NAME
  vector_store = Chroma(
      embedding_function=embeddings,
      collection_name=collection_name,
      persist_directory=CHROMA_DIR
  )
  return vector_store


def ensure_vector_store(transcript: str, collection_name: str | None = None) -> Chroma:
  collection_name = collection_name or get_collection_name(transcript)
  try:
    vector_store = load_vector_store(collection_name)
    vector_store.get(limit=1)
    return vector_store
  except Exception:
    return build_vector_store(transcript, collection_name)


def get_retriever(vector_store: Chroma, k: int = 10):
  return vector_store.as_retriever(
    search_kwargs={"k": k}
  )



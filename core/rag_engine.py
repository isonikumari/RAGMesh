import os
import re

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

from core.vector_store import build_vector_store, get_retriever, get_collection_name, ensure_vector_store
from core.mistral import get_mistral_llm, is_rate_limit_error

def get_llm():
  return get_mistral_llm(temperature=0.4)

def format_docs(docs):
  return "\n\n".join([doc.page_content for doc in docs ])


def sanitize_meeting_context(text: str) -> str:
  if not text:
    return text

  cleaned_lines = []
  for raw_line in re.split(r"(?<=[.!?])\s+|\n+", str(text)):
    line = raw_line.strip()
    if not line:
      continue

    lower = line.lower()
    social_markers = [
      "look fat", "fat in this outfit", "you are ugly", "you are pretty",
      "prettier", "attractive", "handsome", "beauty", "appearance",
      "do i look", "who is prettier"
    ]

    if any(marker in lower for marker in social_markers):
      continue

    cleaned_lines.append(line)

  return " ".join(cleaned_lines).strip() or text.strip()


def build_rag_chain(transcript:str):

  cleaned_transcript = sanitize_meeting_context(transcript)
  vector_store=build_vector_store(cleaned_transcript)
  retriever=get_retriever(vector_store,k=10)
  llm=get_llm()

  prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert meeting assistant. Answer the user's question using only the business-relevant meeting summary/context provided below.
      Ignore unrelated personal chatter, appearance comments, insults, gossip, or social remarks that are not part of the business discussion.
      If the answer is a number, date, amount, or percentage, return the exact value from the summary without replacing it with X or a generic placeholder.
      If the answer is not present in the provided summary, say it is not present in the current summary.
      Always be concise and precise. If quoting someone, mention it clearly.
      Context from the meeting summary:
      {context}"""),
    ("human", "{question}")
  ])


  rag_chain = (
    {"context": retriever | RunnableLambda(format_docs),
     "question": RunnablePassthrough()} | prompt | llm | StrOutputParser()
  )

  return rag_chain

def load_rag_chain(transcript: str | None = None):
  if transcript is None:
    raise ValueError("A transcript is required to load the RAG chain.")

  cleaned_transcript = sanitize_meeting_context(transcript)
  collection_name = get_collection_name(cleaned_transcript)
  vector_store = ensure_vector_store(cleaned_transcript, collection_name)
  retriever = get_retriever(vector_store)

  llm = get_llm()

  prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert meeting assistant. Answer the user's question using only the business-relevant meeting summary/context provided below.
      Ignore unrelated personal chatter, appearance comments, insults, gossip, or social remarks that are not part of the business discussion.
      If the answer is a number, date, amount, or percentage, return the exact value from the summary without replacing it with X or a generic placeholder.
      If the answer is not present in the provided summary, say it is not present in the current summary.
      Always be concise and precise. If quoting someone, mention it clearly.
      Context from the meeting summary:
      {context}"""),
    ("human", "{question}")
  ])

  rag_chain = (
    {"context": retriever | RunnableLambda(format_docs),
     "question": RunnablePassthrough()} | prompt | llm | StrOutputParser()
  )
  return rag_chain

def ask_question(rag_chain, question, source_text: str | None = None) -> str:
  print(f"Question: {question}")

  try:
    answer = rag_chain.invoke(question)
  except Exception as exc:
    print(f"RAG query failed: {exc}")
    if is_rate_limit_error(exc):
      return "Gemini is rate-limiting requests. Please wait a minute and try again, or check your Google Gemini API usage and plan."
    return ""

  print(f"Answer: {answer}")
  return answer
   
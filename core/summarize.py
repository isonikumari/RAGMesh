from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter

import os

from core.mistral import get_mistral_llm

def get_llm():
  return get_mistral_llm(temperature=0.0)


def split_transcript(transcript: str) -> list:
  splitter=RecursiveCharacterTextSplitter(
    chunk_size=3000,
    chunk_overlap=200
  )
  return splitter.split_text(transcript)


def summarize(transcript: str) -> str:
  llm = get_llm()

  map_prompt = ChatPromptTemplate.from_messages(
    [
      ("system", "Summarize this portion of the transcript clearly and concisely for video or audio content. Keep exact names, numbers, dates, currencies, percentages, and deadlines exactly as they appear. Do not replace amounts with placeholders like X or generic values. Remove irrelevant personal chatter or gossip unless it is essential to the business discussion."),
      ("human", "{text}")
    ]
  )
  map_chain = map_prompt | llm | StrOutputParser()
  chunks = split_transcript(transcript)
  chunk_summaries = [map_chain.invoke({"text": chunk}) for chunk in chunks]

  combined = "\n\n".join(chunk_summaries)
  combined_prompt = ChatPromptTemplate.from_messages(
    [
      ("system", "You are an expert transcript summarizer. Combine these partial summaries into one final polished summary for video or audio content. "
       "Use neutral, professional language that fits YouTube videos, lectures, podcasts, or uploaded media. "
       "Return only the summary itself as clear bullet points. Do not mention meetings, do not use a greeting, and do not ask follow-up questions. "
       "Preserve exact financial values, dates, deadlines, names, and percentages exactly as stated. Do not replace amounts with placeholders such as X or generic numbers. "
       "Ignore irrelevant personal remarks about appearance, beauty, or insults unless they are directly relevant to the business context. "
       "Do not include text like 'Would you like to expand on...' or any other invitation to continue."),
      ("human", "{text}")
    ]
  )
  combined_chain = combined_prompt | llm | StrOutputParser()

  return combined_chain.invoke({"text": combined})

def generate_title(transcript: str)-> str:
  llm = get_llm()
  title_chain = (
    ChatPromptTemplate.from_messages([
      ("system", "Based on the transcript, generate a short professional title for the video or audio content "
       "(max 8 words). Only return the title, nothing else."),
       ("human","{text}")
    ]) | llm | StrOutputParser()
  )
  return title_chain.invoke({"text": transcript[:2000]})
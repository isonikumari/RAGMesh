from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
import os
from core.mistral import get_mistral_llm

def get_llm():
    return get_mistral_llm(temperature=0.2)

def build_chain(system_prompt:str):
    llm=get_llm()
    return(
        RunnablePassthrough() | RunnableLambda(lambda x:{"text" : x}) |ChatPromptTemplate.from_messages([
                        ("system", system_prompt),
                        ("human", "{text}")
                ]) |llm | StrOutputParser()
                )

def extract_action_items(transcript:str)->str:
     chain = build_chain(
         "You are a expert meeting analyst. FRom the meeting transcript,"
         "extract all action itens .For each provide :\n"
         "-TAsk description\n"
         "-Owner (who is responsible)\n"
         "-Deadline (if mentioned,else write 'Not Specified)\n\n"
         "Format as a numbered list. If none found say 'No action items found."
     )

     return chain.invoke(transcript)

def extract_question(transcript:str)-> str:
    chain= build_chain(
        "from the meeting transcript,extract all unsolved question"
        "or topics needing follow-up. Format as a numbered list."
        "If none found say 'no open question found ."

    )
    return chain.invoke(transcript)

def extract_key_decisions(transcript:str)->str:
    chain=build_chain(
        "You are a meeting analyst. From the meeting transcript,"
        "extract all key decisions made . Format as a numbered list."
        "If None found say 'No open questions found "
    )
    return chain.invoke(transcript)
# from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama

import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")


# Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful chatbot that can answer questions about \
            everything. You are friendly and knowledgeable.",
        ),
        ("user", "Question:{question}?"),
    ]
)

# Streamlit App
st.title("Local Llama v0.2 Langchain Chatbot")
input_text = st.text_input("Explore the topic you want")

# Ollama Llama Model
llm = Ollama(model="mistral:v0.2")
output_parser = StrOutputParser()
chain = prompt | llm | output_parser

if input_text:
    st.write(chain.invoke({"question": input_text}))

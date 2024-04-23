import requests
import streamlit as st


def get_llama_response(input_text):
    response = requests.post(
        "http://localhost:8000/essay/invoke",
        json={"input": {"topic": input_text}},
    )
    return response.json()["output"]


def get_qwen_response(input_text):
    response = requests.post(
        "http://localhost:8000/poem/invoke",
        json={"input": {"topic": input_text}},
    )
    return response.json()["output"]


st.title("Langchain Demo with Use of APIs")
essay_topic = st.text_input("Enter the topic you want to write an essay on")
poem_topic = st.text_input("Enter the topic you want to write a poem on")

if essay_topic:
    st.write(get_llama_response(essay_topic))

if poem_topic:
    st.write(get_qwen_response(poem_topic))

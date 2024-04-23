from fastapi import FastAPI
from langchain.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langserve import add_routes
import uvicorn
import os
from langchain_community.llms import Ollama

from dotenv import load_dotenv

load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

app = FastAPI(
    title="Langchain Server",
    version="0.1",
    description="A simple API based server using Langchain",
)

# add_routes(app, ChatOpenAI(), path="/llama2")
# add_routes(app, ChatOpenAI(), path="/deepseek-coder:6.7b")
# model = ChatOpenAI()
mistral_llm = Ollama(model="mistral:v0.2")
qwen_llm = Ollama(model="qwen:14b")


# Prompts
prompt1 = ChatPromptTemplate.from_template(
    "You are a professional & Knowledgeable Indian essay writer that \
        writes essays on {topic} in 1K words"
)

prompt2 = ChatPromptTemplate.from_template(
    "You are a professional & Knowledgeable Indian poem writer that \
        writes poems on {topic} in 9 lines"
)

add_routes(app, prompt1 | mistral_llm, path="/essay")
add_routes(app, prompt2 | qwen_llm, path="/poem")

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)

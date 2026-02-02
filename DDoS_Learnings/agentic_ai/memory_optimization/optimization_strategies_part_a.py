# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "marimo>=0.19.0",
#     "pyzmq",
# ]
# ///

import marimo

__generated_with = "0.19.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return


@app.cell
def _():
    import os

    from dotenv import load_dotenv

    load_dotenv()
    return


@app.cell
def _(MessagesState, chat_llm_node):
    from langgraph.graph import StateGraph, START, END
    from langgraph.checkpoint.memory import InMemorySaver

    checkpointer = InMemorySaver()

    builder = StateGraph(MessagesState)
    builder.add_node("chat_llm", chat_llm_node)
    builder.add_edge(START, "chat_llm")
    builder.add_edge("chat_llm", END)

    graph = builder.compile(checkpointer=checkpointer)
    return (graph,)


@app.cell
def _(graph):
    from langchain_core.messages import HumanMessage

    config = {"configurable": {"thread_id": "ticket-seq"}}

    state1 = {
        "messages": [
            HumanMessage(content=(
                "Hi, I'm being charged twice for my subscription. "
                "Can you help me figure out what's going on?"
            ))
        ]
    }
    response1 = graph.invoke(state1, config=config)
    return


if __name__ == "__main__":
    app.run()

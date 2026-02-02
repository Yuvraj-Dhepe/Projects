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
    return (mo,)


@app.cell
def _():
    import os

    from dotenv import load_dotenv

    load_dotenv()
    return (os,)


@app.cell
def _(mo):
    mo.md(r"""
    ### Langchain Basics
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### State
    - Every LangGraph workflow revolves around a **single shared state object**, which is like an **Agent's workspace** that it keeps adding to, reading from, and modifying as it thinks, invokes tools, and responds to the user.
    - It holds everything the Agent knows at any point in time.
    """)
    return


@app.cell
def _():
    from typing_extensions import TypedDict


    class CounterState(TypedDict):
        count: int
    return CounterState, TypedDict


@app.cell
def _(mo):
    mo.md(r"""
    ### Nodes
    - A node in LangGraph is a Python function that implements any executable logic for our agent
    """)
    return


@app.cell
def _(CounterState):
    def increment(state: CounterState) -> dict:
        state["count"] += 1
        return state
    return (increment,)


@app.cell
def _(mo):
    mo.md(r"""
    ### Edges
    - Edges control the  execution flow and define how different nodes communicate with each other
    - Edges can help to express loops, branches or the execution flow for the agents
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Building the Graph
    - We have to create a graph, conforming to a state
    - Add nodes, edges to the graph
    - Point out where to start and where to end
    """)
    return


@app.cell
def _(CounterState, increment):
    from langgraph.graph import END, START, StateGraph

    builder = StateGraph(CounterState)

    builder.add_node("increment", increment)

    # Define the execution order: START -> increment -> END
    builder.add_edge(START, "increment")
    builder.add_edge("increment", END)

    graph = builder.compile()
    return END, START, StateGraph, graph


@app.cell
def _(mo):
    mo.md(r"""
    ### Running the Graph
    - LangGraph, will invoke the graph, define the initial state
    - The state gets modified while traversing through the nodes, following the flow via the edges
    - Execution starts at the START node and ends at END node
    """)
    return


@app.cell
def _(CounterState, graph):
    initial_state: CounterState = {"count": 0}

    result = graph.invoke(initial_state)

    print(result)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Conditional Edges
    - Conditional edges help to route the flow to **one or more** nodes based on **a condition**
    """)
    return


@app.cell
def _(CounterState, END):
    from typing import Literal


    # The should continue node, has to return node names
    def should_continue(state: CounterState) -> Literal["increment", END]:
        if state["count"] < 3:  # keep looping
            return "increment"

        return END  # stop the graph
    return (should_continue,)


@app.cell
def _(CounterState, END, START, StateGraph, increment, should_continue):
    conditional_builder = StateGraph(CounterState)

    conditional_builder.add_node("increment", increment)

    conditional_builder.add_edge(START, "increment")

    conditional_builder.add_conditional_edges(
        "increment",  # Source node of the conditional edge
        should_continue,  # routing function to invoke
        ["increment", END],  # routes to chose from based on the should continue node
    )

    conditional_graph = conditional_builder.compile()

    conditional_result = conditional_graph.invoke({"count": 0})

    print(conditional_result)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Baseline Agent (No Memory)
    - Setting up a baseline agent, to know the impact of each memory technique obvious
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### S1: Define a state
    """)
    return


@app.cell
def _(TypedDict):
    class AgentState(TypedDict):
        user_input: str
        response: str
    return (AgentState,)


@app.cell
def _(mo):
    mo.md(r"""
    #### S2: Define the LLM Node
    - It will accept the state, forms a list of messages with a system prompt and the user input from the state
    """)
    return


@app.cell
def _(AgentState, os):
    from langchain_openai import ChatOpenAI

    baseline_agent = ChatOpenAI(
        model="x-ai/grok-4.1-fast",
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        openai_api_base="https://openrouter.ai/api/v1",
    )

    # S3: Define the LLM Node,

    from langchain_core.messages import HumanMessage, SystemMessage


    def baseline_llm_node(state: AgentState) -> dict:
        messages = [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content=state["user_input"]),
        ]

        reply = baseline_agent.invoke(messages)

        return {"response": reply.content}
    return HumanMessage, SystemMessage, baseline_agent, baseline_llm_node


@app.cell
def _(AgentState, END, START, StateGraph, baseline_llm_node):
    baseline_agent_builder = StateGraph(AgentState)

    baseline_agent_builder.add_node("llm", baseline_llm_node)

    # define the flow: START -> llm -> END
    baseline_agent_builder.add_edge(START, "llm")
    baseline_agent_builder.add_edge("llm", END)

    baseline_graph = baseline_agent_builder.compile()
    return (baseline_graph,)


@app.cell
def _(AgentState, baseline_graph):
    baseline_initial_state: AgentState = {
        "user_input": "What is GIL in python?",
        "response": "",
    }

    baseline_result = baseline_graph.invoke(baseline_initial_state)

    print("User:", baseline_initial_state["user_input"])
    print("Assistant:", baseline_result["response"])
    return


@app.cell
def _(AgentState, baseline_graph):
    baseline_state2: AgentState = {
        "user_input": "Summarise it 2 lines?",
        "response": "",
    }
    baseline_result2 = baseline_graph.invoke(baseline_state2)

    print("\nTurn 2 - User:", baseline_state2["user_input"])
    print("Turn 2 - Assistant:", baseline_result2["response"])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Agents with Memory
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### Checkpoints Demo
    """)
    return


@app.cell
def _(TypedDict):
    from operator import add

    from langgraph.checkpoint.memory import InMemorySaver
    from typing_extensions import Annotated


    class State(TypedDict):
        foo: str
        bar: Annotated[list[str], add]


    def node_a(state: State):
        # overwrite foo, append "a" to bar
        return {"foo": "a", "bar": ["a"]}


    def node_b(state: State):
        # overwrite foo, append "b" to bar
        return {"foo": "b", "bar": ["b"]}
    return Annotated, InMemorySaver, State, add, node_a, node_b


@app.cell
def _(END, InMemorySaver, START, State, StateGraph, node_a, node_b):
    cp_builder = StateGraph(State)
    cp_builder.add_node("node_a", node_a)
    cp_builder.add_node("node_b", node_b)

    cp_builder.add_edge(START, "node_a")
    cp_builder.add_edge("node_a", "node_b")
    cp_builder.add_edge("node_b", END)

    cp_checkpointer = InMemorySaver()
    cp_graph = cp_builder.compile(checkpointer=cp_checkpointer)
    return (cp_graph,)


@app.cell
def _(cp_graph):
    cp_config = {"configurable": {"thread_id": "1"}}

    cp_final_state = cp_graph.invoke({"foo": "", "bar": []}, config=cp_config)
    print("Final state:", cp_final_state)
    return (cp_config,)


@app.cell
def _(cp_config, cp_graph):
    cp_history = list(cp_graph.get_state_history(cp_config))

    for i, snap in enumerate(cp_history[::-1]):
        print(f"\nCheckpoint {i}:")
        print("  created_at:", snap.created_at)
        print("  node:", snap.metadata)
        print("  values:", snap.values)
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### Agents Memory Layer
    """)
    return


@app.cell
def _(Annotated, TypedDict, add):
    from langchain_core.messages import AnyMessage


    class MessagesState(TypedDict):
        messages: Annotated[list[AnyMessage], add]
    return (MessagesState,)


@app.cell
def _(MessagesState, SystemMessage, baseline_agent):
    def chat_llm_node(state: MessagesState):
        history = [SystemMessage(content="You are a customer support assistant.")]
        history.extend(state["messages"])

        reply = baseline_agent.invoke(history)

        return {"messages": [reply]}
    return (chat_llm_node,)


@app.cell
def _(END, InMemorySaver, MessagesState, START, StateGraph, chat_llm_node):
    mem1_checkpointer = InMemorySaver()

    mem1_builder = StateGraph(MessagesState)
    mem1_builder.add_node("chat_llm", chat_llm_node)
    mem1_builder.add_edge(START, "chat_llm")
    mem1_builder.add_edge("chat_llm", END)

    mem1_graph = mem1_builder.compile(checkpointer=mem1_checkpointer)
    return (mem1_graph,)


@app.cell
def _():
    import uuid_utils as uuid


    def generate_uuid():
        return uuid.uuid7()
    return generate_uuid, uuid


@app.cell
def _(generate_uuid):
    u1 = generate_uuid()
    mem1_config = {"configurable": {"thread_id": f"{u1}_{u1.time}"}}
    return (mem1_config,)


@app.cell
def _(HumanMessage, mem1_config, mem1_graph):
    # Turn 1
    mem1_state1 = {
        "messages": [
            HumanMessage(
                content="What is GIL in python? Explain in 3 brief sentences only."
            )
        ]
    }
    mem1_result1 = mem1_graph.invoke(mem1_state1, config=mem1_config)

    for m in mem1_result1["messages"]:
        print(type(m).__name__, ":", m.content)
    return


@app.cell
def _(HumanMessage, mem1_config, mem1_graph):
    # Turn 2
    mem1_state2 = {
        "messages": [HumanMessage(content="Summarise it in 1 line")],
    }
    mem1_result2 = mem1_graph.invoke(
        mem1_state2, config=mem1_config
    )  # We are using same config and same graph

    for mem1_r2 in mem1_result2["messages"][-2:]:
        print(type(mem1_r2).__name__, ":", mem1_r2.content)
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### Inspecting Memory
    """)
    return


@app.cell
def _(mem1_config, mem1_graph):
    # latest state
    mem1_snapshot = mem1_graph.get_state(mem1_config)
    print(mem1_snapshot.values)

    # full history
    mem1_history = list(mem1_graph.get_state_history(mem1_config))
    for mem1_iter, mem1_snap in enumerate(mem1_history[::-1]):
        print(f"\nCheckpoint {mem1_iter}:")
        print("  created_at:", mem1_snap.created_at)
        print("  node:", mem1_snap.metadata)
        print("  values:", mem1_snap.values)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Long Term Memory
    - Long term memories are stored in stores.
    - Stores survive across sessions.
    - Stores are distinguised via namespaces. A namespace acts like a folder or a label that groups related memory items together.
    """)
    return


@app.cell
def _(generate_uuid, uuid):
    from langgraph.store.memory import InMemoryStore

    lt_store = InMemoryStore()

    lt_uid = generate_uuid()
    lt_ns = (str(lt_uid), "memories")

    lt_memory_id = str(uuid.uuid4())
    lt_memory = {"food_preference": "I like pizza"}
    lt_store.put(lt_ns, lt_memory_id, lt_memory)

    lt_memories = lt_store.search(lt_ns)
    lt_latest = lt_memories[-1].dict()
    print(lt_latest["value"])

    # Output
    # {'food_preference': 'I like pizza'}
    return


if __name__ == "__main__":
    app.run()

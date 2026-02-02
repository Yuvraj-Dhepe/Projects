import os
from enum import Enum
from operator import add
from typing import List, Optional

import uuid_utils as uuid
from dotenv import load_dotenv
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage

# LangChain / LangGraph imports
from langchain_openai import ChatOpenAI

# Langfuse import
from langfuse.langchain import CallbackHandler
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.store.memory import InMemoryStore
from pydantic import BaseModel, Field
from typing_extensions import Annotated, TypedDict

# Load environment variables
load_dotenv()

# --- Configuration ---
MODEL_NAME = "google/gemini-3-flash-preview"
USER_ID = str(uuid.uuid4())  # Single static user ID for this session
THREAD_ID = "thread-1"  # Static thread ID for conversation history
MAX_RETRIES = 3  # Maximum number of retries for LLM calls

# --- Langfuse Setup ---
# Initialize the handler. It will automatically pick up keys from os.environ
langfuse_handler = CallbackHandler()

# --- LLM Initialization ---
# --- LLM Initialization ---
llm = ChatOpenAI(
    model=MODEL_NAME,
    temperature=0,
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1",
)

# --- Memory Stores ---
stm_checkpointer = InMemorySaver()
ltm_store = InMemoryStore()

# --- Pydantic Schemas ---


class NamespaceEnum(str, Enum):
    PERSONAL = "personal_life_preferences"
    WORK = "work_life_preferences"


class NamespaceSelection(BaseModel):
    category: NamespaceEnum = Field(
        description="The namespace category that best fits the user's current input."
    )
    reasoning: str = Field(description="Brief reason for selecting this category.")


class MemoryFact(BaseModel):
    key: str = Field(description="The concept name (e.g., 'favorite_food').")
    value: str = Field(description="The specific preference (e.g., 'Italian cuisine').")


class MemoryExtraction(BaseModel):
    should_persist: bool = Field(
        description="True if the message contains new user preferences."
    )
    category: Optional[NamespaceEnum] = Field(
        description="The category where this memory belongs."
    )
    facts: List[MemoryFact] = Field(
        default_factory=list, description="List of key-value pairs to store."
    )


# --- Graph State ---


class AgentState(TypedDict):
    messages: Annotated[List[AnyMessage], add]
    retrieved_context: Optional[str]


# --- Nodes ---
def retrieve_context_node(state: AgentState):
    last_message = state["messages"][-1]

    # 1. Determine Namespace
    structured_llm = llm.with_structured_output(NamespaceSelection)
    system_prompt = (
        "You are a retrieval router. Analyze the user's input and decide which "
        "memory namespace is most relevant for retrieval: 'personal_life_preferences' or 'work_life_preferences'."
    )

    classification = None
    for attempt in range(MAX_RETRIES):
        try:
            classification = structured_llm.invoke(
                [SystemMessage(content=system_prompt), last_message]
            )
            break
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                raise
            print(f"LLM retry {attempt + 1} failed in retrieve_context_node: {e}")

    target_namespace = (USER_ID, classification.category.value)

    # 2. Retrieve from Store
    memories = ltm_store.search(target_namespace)

    context_str = ""
    if memories:
        formatted_memories = [f"{m.value['key']}: {m.value['value']}" for m in memories]
        context_str = f"Context from {classification.category.value}:\n" + "\n".join(
            formatted_memories
        )
        print(
            f"\n[System] Retrieved from {classification.category.value}: {len(memories)} facts found."
        )
    else:
        print(
            f"\n[System] Retrieved from {classification.category.value}: No existing memories."
        )

    return {"retrieved_context": context_str}


def generate_response_node(state: AgentState):
    messages = state["messages"]
    context = state.get("retrieved_context", "")

    system_prompt = "You are a helpful assistant."
    if context:
        system_prompt += f"\n\nRelevant Information about the user:\n{context}"

    prompt_messages = [SystemMessage(content=system_prompt)] + messages

    # The callback handler passed in config will automatically trace this call
    response = llm.invoke(prompt_messages)

    return {"messages": [response]}


def persist_memory_node(state: AgentState):
    last_human_message = None
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            last_human_message = msg
            break

    if not last_human_message:
        return {}

    structured_llm = llm.with_structured_output(MemoryExtraction)
    system_prompt = (
        "You are a memory archivist. Analyze the user's message. "
        "If the user explicitly states a preference or fact about themselves, extract it. "
        "Output a Key-Value pair. Classify it strictly into 'personal_life_preferences' or 'work_life_preferences'."
    )

    extraction = None
    for attempt in range(MAX_RETRIES):
        try:
            extraction = structured_llm.invoke(
                [SystemMessage(content=system_prompt), last_human_message]
            )
            break
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                raise
            print(f"LLM retry {attempt + 1} failed in persist_memory_node: {e}")

    if extraction.should_persist and extraction.facts:
        namespace_tuple = (USER_ID, extraction.category.value)

        for fact in extraction.facts:
            memory_id = str(uuid.uuid4())
            memory_data = {"key": fact.key, "value": fact.value}

            ltm_store.put(namespace_tuple, memory_id, memory_data)
            print(
                f"[System] Memory Saved in '{extraction.category.value}': {fact.key} -> {fact.value}"
            )

    return {}


# --- Graph Construction ---

workflow = StateGraph(AgentState)

workflow.add_node("retrieve", retrieve_context_node)
workflow.add_node("respond", generate_response_node)
workflow.add_node("memorize", persist_memory_node)

workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "respond")
workflow.add_edge("respond", "memorize")
workflow.add_edge("memorize", END)

app = workflow.compile(checkpointer=stm_checkpointer, store=ltm_store)

# --- Execution Simulation ---


def run_interaction(user_input: str):
    # We inject the Langfuse handler into the config
    config = {"configurable": {"thread_id": THREAD_ID}, "callbacks": [langfuse_handler]}

    print("-" * 50)
    print(f"USER: {user_input}")

    initial_state = {"messages": [HumanMessage(content=user_input)]}

    # The 'callbacks' in config are propagated to every node and LLM call within the graph
    result = app.invoke(initial_state, config=config)

    last_msg = result["messages"][-1]
    print(f"AI:   {last_msg.content}")


# --- Scenario Execution ---

print(f"=== Starting Session for User ID: {USER_ID} ===\n")

# 1. User sets Personal Preferences
run_interaction("I love eating Sushi and my favorite color is dark blue.")

# 2. User sets Work Preferences
run_interaction("I work as a Backend Engineer and I prefer coding in Python and Go.")

# 3. Retrieval Challenge: Work Context
run_interaction("Can you recommend a project idea based on my tech stack?")

# 4. Retrieval Challenge: Personal Context
run_interaction(
    "I'm hungry and want to buy a shirt. What suggests do you have based on what I like?"
)

# --- Final Inspection ---

print("\n" + "=" * 50)
print("FINAL MEMORY DUMP")
print("=" * 50)

for cat in [NamespaceEnum.PERSONAL, NamespaceEnum.WORK]:
    ns = (USER_ID, cat.value)
    items = ltm_store.search(ns)
    print(f"\nCategory: {cat.value}")
    if not items:
        print("  (Empty)")
    for item in items:
        data = item.value
        print(f"  - [{data['key']}]: {data['value']}")

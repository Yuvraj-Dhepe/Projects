import marimo

__generated_with = "0.15.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import ollama
    import os
    from raggie import utils

    QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
    QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
    OLLAMA_BASE_URL = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    return mo, ollama, utils


@app.cell
def _(mo):
    mo.md(r"""## Multi-Modal Prompting""")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Conversational LLM
    - When interacting with LLM Models, we interact in form of structured inputs called messages.
    - The messages often include a role and content to specify the **nature of the interaction** and the **information being exchanged**.

    - There are 3 roles:
        - User: This represents the input or query provided by the user interacting with the system.
        - System: This sets the behaviour or persona of the AI model. The role is often used to prime the model for specific task or tones.
        - Assistant: This represents the AI's response to the user query based on the system's instructions and user inputs.

    - Generally in continuous chat systems, for the LLM to keep context of previous message, all the previous user query and the LLMs response are kept intact. That's the reason, how model answers to question relevant to previous questions.
    """
    )
    return


@app.cell
def _(ollama):
    import yfinance as yf


    def get_stock_price(ticker: str) -> float:
        stock = yf.Ticker(ticker)
        return stock.history(period="1d")["Close"].iloc[-1]


    available_functions = {
        "get_stock_price": get_stock_price,
    }


    class Conversation:
        def __init__(self, system=""):
            self.messages = []  # history list
            if system:
                self.messages.append({"role": "system", "content": system})

        def generate(self, user_question, image_path=None, tools=None):
            # Append user query to history under "user" role
            if image_path:
                self.messages.append(
                    {
                        "role": "user",
                        "content": user_question,
                        "images": [image_path],
                    }
                )
            else:
                self.messages.append(
                    {
                        "role": "user",
                        "content": user_question,
                    }
                )

            # Ask model for response (with or without tools)
            if tools:
                response = ollama.chat(
                    model="llama3.2:1b", messages=self.messages, tools=[tools]
                )
            else:
                response = ollama.chat(
                    model="llama3.2-vision", messages=self.messages
                )

            # Handle tool calls if present
            if (
                hasattr(response.message, "tool_calls")
                and response.message.tool_calls
            ):
                for tool in response.message.tool_calls:
                    function_to_call = available_functions.get(tool.function.name)

                    if function_to_call:
                        print("Arguments:", tool.function.arguments)

                        # Call the tool with unpacked arguments
                        result = function_to_call(**tool.function.arguments)
                        print("Function output:", result)

                        # Store the tool call + result in history
                        self.messages.append(
                            {
                                "role": "tool",
                                "name": tool.function.name,
                                "content": str(result),
                            }
                        )

                    else:
                        print("Function not found:", tool.function.name)

            # Add LLM’s textual response to history
            self.messages.append(
                {"role": "assistant", "content": response.message.content}
            )

            return response
    return Conversation, get_stock_price


@app.cell
def _(Conversation, utils):
    system_message = "Your are a teerse expert in high fantasy literature."
    conv = Conversation(system_message)

    # generate the response from the query
    response = conv.generate("Who wrote the book Lord of the Rings?")

    # display the response
    utils.pretty_print(response.message.content)
    return


@app.cell
def _(mo):
    mo.md(r"""### Specify Image in Prompts""")
    return


@app.cell
def _(Conversation, utils):
    img_conv = Conversation("You are a helpful assistant")

    # generate the response from the query
    img_response = img_conv.generate(
        "Describe the image in detail", image_path="./data/dog_img.png"
    )
    # display the response
    utils.pretty_print(img_response.message.content)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Tool Calling 
    - As we see above, with tools like Ollama and multi-modal llms one can work with different modalities, in their continuous conversations.
    - However, sometimes the context required to answer an user-query might not be locally available, in form of files/image etc.
        - For example fetching real-time temperature in a hotel and based on the temperature recommending drinks to users.
        - For such type of tasks, the LLM should know when, it needs some real-time information and via what mediums can it get it.
        - These real-time mediums/edendpoints, which basically add context to the LLM Model, beyond their built-in capabilities are called tools.

    - Such a system with external tools go through the following steps:
        - Recognize whether a task requires external assistance
        - Invoke the appropriate tool or API for that task
        - Process the tools' output and integrte it into the response

    - In such a scenario, AI acts more like a coordinator, delegating tasks it cannot handle internally to specialized tools.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""#### Tool Calling Example""")
    return


@app.cell
def _(Conversation, get_stock_price, utils):
    stock_convo = Conversation("You are a helpful assistant")
    # generate the response from the query
    stock_response = stock_convo.generate(
        "What is the stock price of Apple?", tools=get_stock_price
    )
    # display the response
    utils.pretty_print(stock_response.message.content)
    return (stock_response,)


@app.cell
def _(stock_response):
    stock_response# Notice that the message key in the above response object includes tool_calls along with the argument.
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    - Above tool_calls attribute contains the relevant details, such as:
        - tool.function.name: The name of the tool to be called.
        - tool.function.arguments: The arguments required by the tool.
    """
    )
    return


if __name__ == "__main__":
    app.run()

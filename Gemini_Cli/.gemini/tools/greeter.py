import json
import sys


def say_hello(name: str):
    """Greets the user.

    Args:
        name: The name of the user to greet.
    """
    return f"Hello, {name}!"


if __name__ == "__main__":
    tool_schema = {
        "name": "greeter",
        "description": "A simple tool to greet users.",
        "tools": [
            {
                "name": "say_hello",
                "description": "Greets the user.",
                "args": {
                    "name": {
                        "type": "string",
                        "description": "The name of the user to greet.",
                    }
                },
            }
        ],
    }

    if len(sys.argv) > 1 and sys.argv[1] == "discover":
        print(json.dumps(tool_schema))
    else:
        # Simple execution for demonstration
        request = json.load(sys.stdin)
        tool_name = request["tool"]
        args = request["args"]
        if tool_name == "say_hello":
            result = say_hello(**args)
            print(json.dumps({"result": result}))

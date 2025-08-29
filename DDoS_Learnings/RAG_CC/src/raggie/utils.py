from IPython.display import Markdown, display


def pretty_print(data):
    if isinstance(data, str):
        display(Markdown(data))
    elif isinstance(data, dict):
        for key, value in data.items():
            display(Markdown(f"**{key}:** {value}"))
    else:
        display(data)


def batch_iterate(lst, batch_size):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), batch_size):
        yield lst[i : i + batch_size]

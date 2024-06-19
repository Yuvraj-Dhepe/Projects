import torch
from typing import Callable, Dict, Any
import pandas as pd


def no_grad(func: Callable) -> Callable:
    """
    Decorator to temporarily disable gradient computation.

    This decorator is used to wrap functions or methods that should be executed
    without gradient tracking. It ensures that the function `func` is executed
    within a `torch.no_grad()` context, where gradients are not computed.

    @param Callable func: The function or method to be decorated.

    @return Callable: The decorated function or method.
    """

    def wrapper_nograd(*args, **kwargs):
        """
        Wrapper function to execute `func` without gradient computation.

        This wrapper function is called when the decorated function is invoked.
        It ensures that the function `func` is executed within a
        `torch.no_grad()`
        context, where gradients are not computed.

        @param *args: Positional arguments passed to the decorated function.
        @param **kwargs: Keyword arguments passed to the decorated function.

        @return Any: The result of executing the decorated function.
        """
        with torch.no_grad():
            return func(*args, **kwargs)

    return wrapper_nograd


class CSVLog:
    """
    Logs data to a CSV file using pandas.
    """

    def __init__(self, filename: str):
        """
        Initialize the CSVLog object.

        @param str filename: The name of the CSV file.
        """
        self._filename = filename
        self._header_written = False

    def log(self, items: Dict[str, Any]) -> None:
        """
        Log items to the CSV file.

        @param Dict[str, Any] items: A dictionary containing the items to log.
        @return None: No return value.
        """
        if not self._header_written:
            # Write header if it hasn't been written yet
            pd.DataFrame(columns=items.keys()).to_csv(
                self._filename, index=False
            )
            self._header_written = True

        # Append items to the CSV file
        pd.DataFrame([items]).to_csv(
            self._filename, mode="a", header=False, index=False
        )

"""Mirrors everything printed to stdout/stderr into a text file as well."""
import sys
from pathlib import Path


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data: str) -> None:
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def tee_stdout_to_file(log_path: Path):
    """Redirects stdout/stderr so console output also gets written to `log_path`.

    Returns the opened log file; call `.close()` on it when done (or just let
    the process exit).
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("w")
    sys.stdout = Tee(sys.__stdout__, log_file)
    sys.stderr = Tee(sys.__stderr__, log_file)
    return log_file

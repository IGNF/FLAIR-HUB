import datetime
import os
import sys
import time
from contextlib import contextmanager
from io import StringIO
from typing import Iterator, TextIO

from flair_hub.utils.distributed import is_global_zero, rank_zero_only


class BootstrapStream:
    """Mirror early startup output to the terminal and retain it for replay."""

    def __init__(self, terminal: TextIO) -> None:
        self.terminal = terminal
        self.buffer = StringIO()
        self.encoding = getattr(terminal, "encoding", "utf-8")

    def write(self, message: str) -> None:
        self.terminal.write(message)
        self.buffer.write(message)

    def flush(self) -> None:
        self.terminal.flush()

    def getvalue(self) -> str:
        return self.buffer.getvalue()

    def isatty(self) -> bool:
        return False


@contextmanager
def status_step(label: str, *, enabled: bool | None = None) -> Iterator[None]:
    """Report one timed startup operation on a single flushed log line."""
    show = is_global_zero() if enabled is None else enabled
    started = time.perf_counter()
    if show:
        print(f"[....] {label:<58}", end="", flush=True)
    try:
        yield
    except BaseException:
        if show:
            print(f" [FAIL] ({time.perf_counter() - started:.2f}s)", flush=True)
        raise
    else:
        if show:
            print(f" [ OK ] ({time.perf_counter() - started:.2f}s)", flush=True)


@rank_zero_only
def start_msg():
    print(r"""
  _____ _        _    ___ ____       _   _ _   _ ____
 |  ___| |      / \  |_ _|  _ \     | | | | | | | __ )
 | |_  | |     / _ \  | || |_) _____| |_| | | | |  _ \
 |  _| | |___ / ___ \ | ||  _ |_____|  _  | |_| | |_) |
 |_|   |_____/_/   \_|___|_| \_\    |_| |_|\___/|____/
_______________________________________________________

=======================================================
                       LAUNCHING
=======================================================""", flush=True)
    print(datetime.datetime.now().strftime("Starting: %Y-%m-%d %H:%M:%S"), flush=True)
    print(flush=True)


@rank_zero_only
def end_msg():
    print("""
=======================================================
                        FINISHED
=======================================================""", flush=True)
    print(datetime.datetime.now().strftime("Finished: %Y-%m-%d %H:%M:%S"), flush=True)
    print(flush=True)


class Logger:
    """
    Custom logger that mirrors stdout to both the terminal and a log file.
    Useful for capturing experiment logs during training or inference.

    Attributes:
        terminal (TextIO): The original system stdout.
        log (TextIO): The file object to write logs to.
    """

    def __init__(self, filename: str = 'Default.log', terminal: TextIO | None = None) -> None:
        """
        Initialize a Logger instance that writes both to stdout and a file.

        Args:
            filename (str): The log file name. Will be made unique if already exists.
        """
        self.filename = self._get_unique_filename(filename)
        self.terminal = terminal if terminal is not None else sys.stdout
        self.log = open(self.filename, 'w', encoding='utf-8')
        self.encoding = getattr(self.terminal, "encoding", "utf-8")

    def _get_unique_filename(self, filename: str) -> str:
        """
        Generate a unique filename by appending a version suffix if needed.

        Args:
            filename (str): The initial filename requested.

        Returns:
            str: A unique filename that doesn't overwrite existing files.
        """
        base, ext = os.path.splitext(filename)
        if not os.path.exists(filename):
            return filename

        version = 1
        while True:
            new_filename = f"{base}_v{version}{ext}"
            if not os.path.exists(new_filename):
                return new_filename
            version += 1

    def write(self, message: str) -> None:
        """
        Write a message to both stdout and the log file.

        Args:
            message (str): The string message to be logged.
        """
        self.terminal.write(message)
        self.log.write(message)

    def flush(self) -> None:
        """
        Flush the log file stream.
        """
        self.terminal.flush()
        self.log.flush()

    def write_log_only(self, message: str) -> None:
        """Write retained bootstrap output to the file without duplicating it."""
        self.log.write(message)
        self.log.flush()

    def close(self) -> None:
        """
        Close the log file stream.
        """
        self.log.close()

    def isatty(self) -> bool:
        """
        Indicates whether the stream is interactive. Needed for tqdm compatibility.

        Returns:
            bool: Always False to ensure tqdm behaves correctly.
        """
        return False


class DiagnosticLogger:
    """Mirror stderr to its original destination and a shared diagnostics file.

    The file is opened in append mode because every distributed rank executes
    the application entry point. This preserves the scheduler/terminal stderr
    stream while collecting Python warnings and tracebacks from all ranks in
    one experiment-local file.
    """

    def __init__(self, filename: str, terminal: TextIO | None = None) -> None:
        self.filename = filename
        self.terminal = terminal if terminal is not None else sys.stderr


        self.log = open(filename, "a", encoding="utf-8")
        self.encoding = getattr(self.terminal, "encoding", "utf-8")

    def write(self, message: str) -> None:
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self) -> None:
        self.terminal.flush()
        self.log.flush()

    def write_log_only(self, message: str) -> None:
        """Append retained early stderr without printing it a second time."""
        self.log.write(message)
        self.log.flush()

    def close(self) -> None:
        self.log.close()

    def isatty(self) -> bool:
        """Preserve the behavior of the original stderr destination."""
        return bool(getattr(self.terminal, "isatty", lambda: False)())

    def fileno(self) -> int:
        """Expose the original descriptor for libraries that inspect stderr."""
        return self.terminal.fileno()

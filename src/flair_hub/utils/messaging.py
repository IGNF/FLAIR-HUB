import sys, os
import datetime

from pytorch_lightning.utilities.rank_zero import rank_zero_only  


@rank_zero_only
def start_msg():
    print("""
  _____ _        _    ___ ____       _   _ _   _ ____  
 |  ___| |      / \  |_ _|  _ \     | | | | | | | __ ) 
 | |_  | |     / _ \  | || |_) _____| |_| | | | |  _ \ 
 |  _| | |___ / ___ \ | ||  _ |_____|  _  | |_| | |_) |
 |_|   |_____/_/   \_|___|_| \_\    |_| |_|\___/|____/ 
_______________________________________________________

#######################################################         
####################  LAUNCHING #######################
    """)
    print(datetime.datetime.now().strftime("Starting: %Y-%m-%d  %H:%M") + '\n')
    print("""
[ ] Setting up Logger     . . .
[ ] Creating output files . . . 
[ ] Reading config files  . . .
[ ] Building up datasets  . . . 

    """
    )


@rank_zero_only
def end_msg():
    print("""
#######################################################         
####################  FINISHED  #######################    
""")
    print(datetime.datetime.now().strftime("Ending: %Y-%m-%d  %H:%M") + '\n')
    


@rank_zero_only
class Logger:
    """
    Custom logger that mirrors stdout to both the terminal and a log file.
    Useful for capturing experiment logs during training or inference.

    Attributes:
        terminal (TextIO): The original system stdout.
        log (TextIO): The file object to write logs to.
    """

    def __init__(self, filename: str = 'Default.log') -> None:
        """
        Initialize a Logger instance that writes both to stdout and a file.

        Args:
            filename (str): The log file name. Will be made unique if already exists.
        """
        filename = self._get_unique_filename(filename)
        self.terminal = sys.stdout
        self.log = open(filename, 'w', encoding='utf-8')
        self.encoding = self.terminal.encoding

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

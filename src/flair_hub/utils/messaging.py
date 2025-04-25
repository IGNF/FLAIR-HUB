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
    


@rank_zero_only
class Logger:
    def __init__(self, filename: str = 'Default.log') -> None:
        """
        Initializes a custom logger to output to both terminal and log file.

        Args:
            filename (str): Name of the log file.
        """
        filename = self._get_unique_filename(filename)
        self.terminal = sys.stdout
        self.log = open(filename, 'w', encoding='utf-8')
        self.encoding = self.terminal.encoding

    def _get_unique_filename(self, filename: str) -> str:
        """
        Checks if the file exists and appends a version number if needed.

        Args:
            filename (str): Initial filename.

        Returns:
            str: Unique filename with version number if necessary.
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
        Writes the log message to both the terminal and the log file.

        Args:
            message (str): The message to be logged.
        """
        self.terminal.write(message)
        self.log.write(message)

    def flush(self) -> None:
        """
        Flushes the log file to ensure all data is written.
        """
        self.log.flush()

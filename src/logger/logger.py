import os
import logging
from typing import Union

from src.utils.constants import BASE_LOG_DIR


class Logger:
    def __init__(
        self, 
        log_file: str = "log.txt", 
        log_level: Union[int, str] = logging.INFO,
        is_stream_handle: bool = True,
        format: str = '%(asctime)s - %(levelname)s - %(message)s'
    ) -> None:
        if isinstance(log_level, str):
            log_level = self.set_string_level_to_int(log_level)

        # Logging properties
        log_file_path = os.path.join(BASE_LOG_DIR, log_file)

        # Create the log directory if it does not exist
        os.makedirs(BASE_LOG_DIR, exist_ok=True)

        with open(log_file_path, "w") as f:
            f.write("")

        handlers: list[logging.Handler] = [logging.FileHandler(log_file_path)]
        if is_stream_handle: handlers.append(logging.StreamHandler())

        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format=format,
            handlers=handlers
        )

        self.logger = logging.getLogger(__name__)    

    def set_string_level_to_int(self, level: str) -> int:
        if level == "info":
            return logging.INFO
        else:
            return logging.DEBUG

    def info(self, content: str) -> None:
        self.logger.info(content)

    def error(self, content: str) -> None:
        self.logger.error(content)
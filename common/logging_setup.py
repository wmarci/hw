import logging
import sys
import os

LOG_FORMAT = (
    "%(asctime)s - %(name)s - %(funcName)s:%(lineno)d - %(levelname)s - %(message)s"
)
FORMATTER = logging.Formatter(LOG_FORMAT)


def configure_logger(config: dict):
    """Root logger configuration.

    :param config: logging config
    :type config: dict
    """

    os.makedirs(os.path.join(os.getenv("STAGE"), "logs"), exist_ok=True)
    root_logger = logging.getLogger()
    root_logger.setLevel(config.get("level"))
    file_handler = logging.FileHandler(
        os.path.join(os.getenv("STAGE"), "logs", config.get("logfile"))
    )
    file_handler.setFormatter(FORMATTER)
    root_logger.addHandler(file_handler)
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(FORMATTER)
    root_logger.addHandler(stdout_handler)

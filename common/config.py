import os
import json
import sys
import logging

logger = logging.getLogger(__name__)


class Config:
    """Obtain configuration"""

    def __init__(self):
        if os.getenv("CONFIG_FILE_PATH") is None:
            logger.error("Configuration file path is missing!")
            sys.exit(0)
        else:
            with open(os.getenv("CONFIG_FILE_PATH"), "r") as json_file:
                self.full_config = json.load(json_file)

    def __getitem__(self, key):
        return self.full_config[key]


config = Config()

import logging
from .config import config

logger = logging.getLogger(__name__)

MAPPING = {v: k for k, v in config["data_prep"].get("mapping").items()}

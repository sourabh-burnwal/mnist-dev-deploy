import logging
import os
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

from config import project_config

PROJECT_NAME = project_config["PROJECT_NAME"]


def configure_logging(log_level=logging.DEBUG):
    current_file_path = Path(__file__)
    os.makedirs(os.path.join(current_file_path.parent.parent, 'logs/'), exist_ok=True)
    log_file_path = os.path.join(current_file_path.parent.parent, "logs/logs.log")
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler = TimedRotatingFileHandler(log_file_path, when="midnight", interval=1, encoding="utf8")
    handler_suffix = "%Y-%m-%d"
    handler.setFormatter(formatter)
    logger = logging.getLogger(PROJECT_NAME)
    logger.setLevel(log_level)
    logger.addHandler(handler)
    logger.info(f'{PROJECT_NAME} startup')

    return logger

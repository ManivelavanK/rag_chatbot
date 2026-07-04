# logger.py — structured logging for all pipeline components

import logging
import os
from datetime import datetime
from config import LOG_DIR

os.makedirs(LOG_DIR, exist_ok=True)

_log_file = os.path.join(LOG_DIR, f"rag_{datetime.now().strftime('%Y%m%d')}.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.FileHandler(_log_file, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)

def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)

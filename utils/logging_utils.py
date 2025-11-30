import json
import logging
from pathlib import Path
from typing import Any, Dict


def setup_file_logger(path: Path) -> logging.Logger:
    logger = logging.getLogger(str(path))
    if logger.handlers:
        return logger
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(path)
    formatter = logging.Formatter("%(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


def log_json(logger: logging.Logger, payload: Dict[str, Any]) -> None:
    try:
        logger.info(json.dumps(payload))
    except Exception:
        pass

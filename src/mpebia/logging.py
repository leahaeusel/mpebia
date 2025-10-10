"""Utility functions related to logging."""

import logging
import sys
from pathlib import Path


def get_logger(file):
    """Get a logger that logs to a file and the console.

    Args:
        file (str): Path of the file that needs the logger.

    Returns:
        Logger: Logger that logs to a file and the console.
    """
    log_file = Path(file).with_suffix(".log")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler(log_file, mode="w")
    fh.setLevel(logging.INFO)
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s: %(levelname)s: %(message)s", datefmt="%H:%M:%S")
    fh.setFormatter(formatter)
    sh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(sh)

    return logger

"""
models/logger.py
----------------
Journalisation simple utilisée par tous les blocs du pipeline
(Import, EDA, Prétraitement, Machine Learning, Export).
"""

from __future__ import annotations

import logging
import os

LOG_FILE = os.path.join("logs", "app.log")


def get_logger(name: str = "pfa") -> logging.Logger:
    """Retourne un logger configuré (fichier + console), créé une seule fois."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)

    return logger

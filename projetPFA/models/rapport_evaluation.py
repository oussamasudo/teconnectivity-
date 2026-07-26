"""
models/rapport_evaluation.py
------------------------------
Classe RapportEvaluation : calcule et présente les métriques
d'évaluation du modèle (accuracy, precision, recall, f1Score,
matriceConfusion), conforme au diagramme de classes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from models.logger import get_logger

logger = get_logger(__name__)


@dataclass
class RapportEvaluation:
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1Score: float = 0.0
    matriceConfusion: np.ndarray = field(default_factory=lambda: np.zeros((2, 2)))

    def calculerMetriques(
        self, y_reel: np.ndarray, y_predit: np.ndarray, labels: list[int] | None = None
    ) -> "RapportEvaluation":
        """Calcule les métriques. `average='weighted'` fonctionne aussi bien
        pour une cible binaire (Non_utile/Utile) que pour un cas à N classes
        si le pipeline est étendu plus tard. `labels` permet de fixer l'ordre
        des classes dans la matrice de confusion (ex. [0, 1] = Non_utile, Utile)."""
        self.accuracy = float(accuracy_score(y_reel, y_predit))
        self.precision = float(
            precision_score(y_reel, y_predit, average="weighted", zero_division=0)
        )
        self.recall = float(
            recall_score(y_reel, y_predit, average="weighted", zero_division=0)
        )
        self.f1Score = float(
            f1_score(y_reel, y_predit, average="weighted", zero_division=0)
        )
        self.matriceConfusion = confusion_matrix(y_reel, y_predit, labels=labels)
        logger.info(
            "Évaluation : accuracy=%.3f precision=%.3f recall=%.3f f1=%.3f",
            self.accuracy, self.precision, self.recall, self.f1Score,
        )
        return self

    def afficherRapport(self) -> str:
        """Retourne un résumé texte du rapport (utilisé en fallback hors Streamlit)."""
        return (
            f"Accuracy  : {self.accuracy:.3f}\n"
            f"Precision : {self.precision:.3f}\n"
            f"Recall    : {self.recall:.3f}\n"
            f"F1-score  : {self.f1Score:.3f}\n"
            f"Matrice de confusion :\n{self.matriceConfusion}"
        )

    def to_dict(self) -> dict:
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1Score": self.f1Score,
        }
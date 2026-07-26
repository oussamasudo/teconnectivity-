"""
utils/metrics.py
-------------------
Fonctions utilitaires pour la synthèse des statistiques du pipeline
(affichées dans les cartes de métriques de app.py).
"""

from __future__ import annotations

import pandas as pd


def resume_pipeline(nb_lignes_brutes: int, nb_valides: int, nb_rejetees: int) -> dict:
    taux = (nb_valides / nb_lignes_brutes * 100) if nb_lignes_brutes else 0.0
    return {
        "nb_lignes_brutes": nb_lignes_brutes,
        "nb_valides": nb_valides,
        "nb_rejetees": nb_rejetees,
        "taux_conservation": round(taux, 1),
    }


def matrice_confusion_vers_dataframe(matrice) -> pd.DataFrame:
    """Transforme une matrice de confusion numpy 2x2 en DataFrame lisible."""
    return pd.DataFrame(
        matrice,
        index=["Réel : Non_utile (0)", "Réel : Utile (1)"],
        columns=["Prédit : Non_utile (0)", "Prédit : Utile (1)"],
    )

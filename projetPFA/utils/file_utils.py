"""
utils/file_utils.py
---------------------
Fonctions utilitaires de gestion de fichiers (upload temporaire, etc.).
"""

from __future__ import annotations

import os
import tempfile


def sauvegarder_upload_temporaire(fichier_uploade, dossier: str = "uploads") -> str:
    """Sauvegarde un fichier uploadé (Streamlit UploadedFile) dans un dossier temporaire
    et retourne le chemin complet du fichier écrit sur disque."""
    os.makedirs(dossier, exist_ok=True)
    chemin = os.path.join(dossier, fichier_uploade.name)
    with open(chemin, "wb") as f:
        f.write(fichier_uploade.getbuffer())
    return chemin


def nettoyer_dossier(dossier: str) -> None:
    """Supprime tous les fichiers d'un dossier (utilisé pour vider uploads/ ou outputs/)."""
    if not os.path.isdir(dossier):
        return
    for nom in os.listdir(dossier):
        chemin = os.path.join(dossier, nom)
        if os.path.isfile(chemin):
            os.remove(chemin)

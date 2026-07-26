"""
utils/excel_utils.py
-----------------------
Fonctions utilitaires pour la lecture/écriture de fichiers Excel/CSV.
"""

from __future__ import annotations

import pandas as pd


def lire_fichier_tabulaire(fichier_uploade) -> pd.DataFrame:
    """Lit un fichier uploadé (.xlsx, .xls ou .csv) et retourne un DataFrame."""
    nom = fichier_uploade.name.lower()
    if nom.endswith(".csv"):
        return pd.read_csv(fichier_uploade)
    return pd.read_excel(fichier_uploade)


def dataframe_vers_bytes_excel(df: pd.DataFrame, nom_feuille: str = "Feuille1") -> bytes:
    """Convertit un DataFrame en bytes .xlsx (pour un bouton de téléchargement Streamlit)."""
    import io

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=nom_feuille)
    return buffer.getvalue()

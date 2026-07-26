"""
models/export_excel.py
------------------------
Bloc 5 - GENERATION DU NOUVEAU FICHIER EXCEL.

Classe ExportExcel : génère et enregistre le fichier Excel propre
(Donnees_Nettoyees.xlsx) — lignes utiles uniquement, colonnes pertinentes
conservées — ainsi qu'un rapport d'audit des lignes rejetées. Ce fichier
alimente ensuite Power BI (hors code, étape 6 du diagramme de séquence).
"""

from __future__ import annotations

import io
import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from config import (
    COLONNE_CIBLE,
    DEFAULT_AUDIT_NAME,
    DEFAULT_OUTPUT_NAME,
    FEUILLE_DONNEES_VALIDES,
    FEUILLE_LIGNES_REJETEES,
    OUTPUT_DIR,
    SEUIL_COLONNE_VIDE_EXPORT,
)
from models.logger import get_logger
from models.modele_ml import CLASSES_UTILITE, generer_labels_utilite

logger = get_logger(__name__)


@dataclass
class ExportExcel:
    cheminSortie: str = OUTPUT_DIR

    # ------------------------------------------------------------------
    def colonnes_pertinentes(
        self, df: pd.DataFrame, seuil_vide: float = SEUIL_COLONNE_VIDE_EXPORT
    ) -> list[str]:
        """Sélectionne, de façon GENERIQUE, les colonnes "pertinentes" à
        conserver dans le fichier final : celles qui ne sont pas quasiment
        entièrement vides (au-delà de `seuil_vide` % de valeurs manquantes),
        ni constantes (une seule valeur distincte sur tout le fichier)."""
        colonnes = []
        n = len(df)
        for col in df.columns:
            if n and df[col].isna().mean() > seuil_vide:
                continue
            if df[col].nunique(dropna=True) <= 1:
                continue
            colonnes.append(col)
        return colonnes or list(df.columns)

    def genererExcel(self, df: pd.DataFrame, nom_feuille: str = "Donnees") -> bytes:
        """Génère un fichier Excel en mémoire (bytes), prêt pour un bouton de téléchargement."""
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name=nom_feuille)
        return buffer.getvalue()

    def enregistrer(
        self, df: pd.DataFrame, nom_fichier: str = DEFAULT_OUTPUT_NAME, nom_feuille: str = "Donnees"
    ) -> str:
        """Enregistre le fichier Excel sur le disque, dans le dossier outputs/."""
        os.makedirs(self.cheminSortie, exist_ok=True)
        chemin_complet = os.path.join(self.cheminSortie, nom_fichier)
        df.to_excel(chemin_complet, index=False, sheet_name=nom_feuille)
        logger.info("Fichier exporté : %s (%d lignes)", chemin_complet, len(df))
        return chemin_complet

    def genererRapportAudit(self, df_rejete: pd.DataFrame) -> bytes:
        return self.genererExcel(df_rejete, nom_feuille=FEUILLE_LIGNES_REJETEES)

    def _filtrer_df_utiles(self, df: pd.DataFrame) -> pd.DataFrame:
        """Retourne le DataFrame nettoyé en ne gardant que les lignes utiles."""
        df = df.dropna(how="all").dropna(axis=1, how="all")
        if df.empty:
            return df

        if COLONNE_CIBLE in df.columns:
            colonne = df[COLONNE_CIBLE].astype(str).str.strip()
            if colonne.isin(CLASSES_UTILITE).any():
                mask = colonne == CLASSES_UTILITE[1]
            else:
                mask = pd.to_numeric(colonne, errors="coerce") == 1
            return df.loc[mask].reset_index(drop=True)

        labels, _ = generer_labels_utilite(df)
        return df.loc[labels.astype(bool)].reset_index(drop=True)

    def genererExportNettoyeDepuisFichier(self, chemin_source: str) -> bytes:
        """Génère un fichier Excel nettoyé en préservant la structure d'origine."""
        buffer = io.BytesIO()
        if chemin_source.lower().endswith(".csv"):
            df = pd.read_csv(chemin_source)
            df_nettoye = self._filtrer_df_utiles(df)
            nom_feuille = Path(chemin_source).stem[:31]
            with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
                df_nettoye.to_excel(writer, index=False, sheet_name=nom_feuille)
            return buffer.getvalue()

        xls = pd.ExcelFile(chemin_source)
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            for feuille in xls.sheet_names:
                df = pd.read_excel(xls, sheet_name=feuille)
                df_nettoye = self._filtrer_df_utiles(df)
                df_nettoye.to_excel(writer, index=False, sheet_name=feuille[:31])
        return buffer.getvalue()

    # ------------------------------------------------------------------
    # Export générique "2 onglets" : lignes utiles (colonnes pertinentes
    # uniquement) + lignes rejetées (audit, toutes colonnes conservées).
    # Remplace l'export "multi-KPI" métier de la version précédente.
    # ------------------------------------------------------------------
    def genererExportComplet(
        self, df_utile: pd.DataFrame, df_rejete: pd.DataFrame
    ) -> bytes:
        colonnes_ok = self.colonnes_pertinentes(df_utile)
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            df_utile[colonnes_ok].to_excel(
                writer, index=False, sheet_name=FEUILLE_DONNEES_VALIDES[:31]
            )
        return buffer.getvalue()

    def enregistrerExportComplet(
        self,
        df_utile: pd.DataFrame,
        df_rejete: pd.DataFrame,
        nom_fichier: str = DEFAULT_OUTPUT_NAME,
    ) -> str:
        os.makedirs(self.cheminSortie, exist_ok=True)
        chemin_complet = os.path.join(self.cheminSortie, nom_fichier)
        colonnes_ok = self.colonnes_pertinentes(df_utile)
        with pd.ExcelWriter(chemin_complet, engine="openpyxl") as writer:
            df_utile[colonnes_ok].to_excel(
                writer, index=False, sheet_name=FEUILLE_DONNEES_VALIDES[:31]
            )
        logger.info(
            "Fichier nettoyé exporté : %s (%d lignes utiles)",
            chemin_complet, len(df_utile),
        )
        return chemin_complet

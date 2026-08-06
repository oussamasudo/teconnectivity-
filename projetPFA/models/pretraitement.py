"""
models/pretraitement.py
-------------------------
Bloc 3 - PRETRAITEMENT.

Classe Pretraitement : nettoyage (doublons, valeurs manquantes),
normalisation / standardisation, encodage des variables catégorielles.
Conforme au diagramme de classes (methodeNormalisation, methodeEncodage,
supprimerDoublons, traiterValeursManquantes, normaliser, standardiser,
encoderVariables).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from config import METHODE_ENCODAGE_DEFAUT, METHODE_NORMALISATION_DEFAUT
from models.jeu_donnees import JeuDonnees
from models.logger import get_logger

logger = get_logger(__name__)


def dedupliquer_colonnes(df: pd.DataFrame) -> pd.DataFrame:
    """Renomme les colonnes en double en leur ajoutant un suffixe numéroté.

    Un encodage One-Hot générique (`pd.get_dummies`) peut produire deux
    colonnes de même nom quand deux colonnes SOURCE différentes, combinées à
    leurs valeurs, donnent le même résultat (ex. colonne "A" valeur "B_x" et
    colonne "A_B" valeur "x" -> "A_B_x" dans les deux cas). scikit-learn
    refuse ensuite d'entraîner un modèle sur un DataFrame à noms de colonnes
    non uniques ; on désambiguïse donc systématiquement plutôt que de
    laisser planter l'entraînement sur un fichier réel imprévisible.
    """
    if not df.columns.duplicated().any():
        return df
    vus: dict[str, int] = {}
    nouvelles_colonnes = []
    for nom in df.columns:
        if nom not in vus:
            vus[nom] = 0
            nouvelles_colonnes.append(nom)
        else:
            vus[nom] += 1
            nouvelles_colonnes.append(f"{nom}__{vus[nom]}")
    df = df.copy()
    df.columns = nouvelles_colonnes
    return df


@dataclass
class Pretraitement:
    methodeNormalisation: str = METHODE_NORMALISATION_DEFAUT  # "standard" | "minmax"
    methodeEncodage: str = METHODE_ENCODAGE_DEFAUT             # "onehot" | "label"
    _nb_doublons_supprimes: int = field(default=0, repr=False)
    _nb_valeurs_imputees: int = field(default=0, repr=False)

    # ------------------------------------------------------------------
    def supprimerDoublons(self, jeu: JeuDonnees) -> JeuDonnees:
        """Supprime les lignes strictement dupliquées.

        Important : l'index d'origine est CONSERVÉ (pas de reset_index) afin
        de pouvoir réaligner les étiquettes (labels) générées avant nettoyage
        sur les lignes qui subsistent après suppression des doublons.
        """
        df = jeu.dataframe
        avant = len(df)
        df = df.drop_duplicates()
        self._nb_doublons_supprimes = avant - len(df)
        logger.info("Doublons supprimés : %d", self._nb_doublons_supprimes)
        return JeuDonnees(idDataset=jeu.idDataset, dataframe=df, dateImport=jeu.dateImport)

    def traiterValeursManquantes(self, jeu: JeuDonnees) -> JeuDonnees:
        """Impute les valeurs manquantes : médiane pour le numérique, mode pour le catégoriel."""
        df = jeu.dataframe.copy()
        nb_imputees = 0
        for col in df.columns:
            nb_na = df[col].isna().sum()
            if nb_na == 0:
                continue
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median())
            else:
                mode = df[col].mode()
                valeur = mode.iloc[0] if not mode.empty else "Inconnu"
                df[col] = df[col].fillna(valeur)
            nb_imputees += int(nb_na)
        self._nb_valeurs_imputees = nb_imputees
        logger.info("Valeurs manquantes imputées : %d", nb_imputees)
        return JeuDonnees(idDataset=jeu.idDataset, dataframe=df, dateImport=jeu.dateImport)

    def normaliser(self, jeu: JeuDonnees, colonnes: list[str] | None = None) -> JeuDonnees:
        """Normalisation Min-Max des colonnes numériques (bornes 0-1)."""
        df = jeu.dataframe.copy()
        # `colonnes` peut être explicitement une liste VIDE (ex. aucune colonne
        # numérique retenue après exclusion des colonnes de fuite/texte libre) :
        # un simple `colonnes or [...]` traiterait ce cas comme un `None` et
        # redéclencherait une auto-détection non voulue par l'appelant. Seul
        # `None` doit déclencher l'auto-détection.
        cols = colonnes if colonnes is not None else list(df.select_dtypes(include=[np.number]).columns)
        if cols:
            scaler = MinMaxScaler()
            df[cols] = scaler.fit_transform(df[cols])
        return JeuDonnees(idDataset=jeu.idDataset, dataframe=df, dateImport=jeu.dateImport)

    def standardiser(self, jeu: JeuDonnees, colonnes: list[str] | None = None) -> JeuDonnees:
        """Standardisation (StandardScaler, moyenne 0 / écart-type 1) des colonnes numériques."""
        df = jeu.dataframe.copy()
        cols = colonnes if colonnes is not None else list(df.select_dtypes(include=[np.number]).columns)
        if cols:
            scaler = StandardScaler()
            df[cols] = scaler.fit_transform(df[cols])
        return JeuDonnees(idDataset=jeu.idDataset, dataframe=df, dateImport=jeu.dateImport)

    def extraireFeaturesDate(self, jeu: JeuDonnees, colonne: str = "Date") -> JeuDonnees:
        """Décompose une colonne datetime en features numériques exploitables
        par le modèle (mois, semaine ISO, jour de semaine), puis supprime la
        colonne Date d'origine (non numérique, non catégorielle utile telle quelle).

        Important : certaines lignes peuvent avoir une Date manquante/invalide
        (NaT), notamment sur un fichier réel partiellement rempli. On garde
        alors ces features en `float` (NaN autorisé) plutôt que de forcer un
        `int`, qui plante sur NaN. L'étape `traiterValeursManquantes`,
        exécutée juste après dans le pipeline, se charge de les imputer.
        """
        df = jeu.dataframe.copy()
        if colonne in df.columns and pd.api.types.is_datetime64_any_dtype(df[colonne]):
            df[f"{colonne}_mois"] = df[colonne].dt.month.astype("float64")
            df[f"{colonne}_semaine"] = df[colonne].dt.isocalendar().week.astype("float64")
            df[f"{colonne}_jour_semaine"] = df[colonne].dt.dayofweek.astype("float64")
            df = df.drop(columns=[colonne])
        return JeuDonnees(idDataset=jeu.idDataset, dataframe=df, dateImport=jeu.dateImport)

    def encoderVariables(self, jeu: JeuDonnees, colonnes: list[str] | None = None) -> JeuDonnees:
        """Encode les variables catégorielles (one-hot par défaut, ou label encoding)."""
        df = jeu.dataframe.copy()
        cols = colonnes or list(df.select_dtypes(include=["object", "category"]).columns)
        if not cols:
            return JeuDonnees(idDataset=jeu.idDataset, dataframe=df, dateImport=jeu.dateImport)

        if self.methodeEncodage == "label":
            for col in cols:
                df[col] = df[col].astype("category").cat.codes
        else:  # onehot
            df = pd.get_dummies(df, columns=cols, drop_first=False)
            df = dedupliquer_colonnes(df)

        return JeuDonnees(idDataset=jeu.idDataset, dataframe=df, dateImport=jeu.dateImport)

    # ------------------------------------------------------------------
    def executer_pipeline_complet(
        self, jeu: JeuDonnees, colonnes_a_ignorer: list[str] | None = None
    ) -> JeuDonnees:
        """Enchaîne les 4 étapes : doublons -> NaN -> normalisation/standardisation -> encodage."""
        ignorees = set(colonnes_a_ignorer or [])
        jeu = self.supprimerDoublons(jeu)
        jeu = self.extraireFeaturesDate(jeu)
        jeu = self.traiterValeursManquantes(jeu)

        colonnes_num = [
            c for c in jeu.dataframe.select_dtypes(include=[np.number]).columns
            if c not in ignorees
        ]
        if self.methodeNormalisation == "minmax":
            jeu = self.normaliser(jeu, colonnes_num)
        else:
            jeu = self.standardiser(jeu, colonnes_num)

        colonnes_cat = [
            c for c in jeu.dataframe.select_dtypes(include=["object", "category"]).columns
            if c not in ignorees
        ]
        jeu = self.encoderVariables(jeu, colonnes_cat)
        return jeu

    @property
    def rapport(self) -> dict:
        return {
            "doublons_supprimes": self._nb_doublons_supprimes,
            "valeurs_imputees": self._nb_valeurs_imputees,
        }
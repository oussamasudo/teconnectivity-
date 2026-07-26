"""
models/jeu_donnees.py
----------------------
Classe JeuDonnees : encapsule un DataFrame pandas et expose les mêmes
attributs/méthodes que dans le diagramme de classes (idDataset,
nombreLignes, nombreColonnes, colonnes, dateImport, ajouterColonne,
supprimerColonne, getColonnes, getNbLignes, getNbColonnes).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

import pandas as pd


@dataclass
class JeuDonnees:
    idDataset: int
    dataframe: pd.DataFrame
    dateImport: date = field(default_factory=date.today)

    # -- attributs dérivés (recalculés à la volée pour rester synchrones) --
    @property
    def nombreLignes(self) -> int:
        return len(self.dataframe)

    @property
    def nombreColonnes(self) -> int:
        return len(self.dataframe.columns)

    @property
    def colonnes(self) -> list[str]:
        return list(self.dataframe.columns)

    # ------------------------------------------------------------------
    def ajouterColonne(self, nom: str, valeurs: Any) -> None:
        """Ajoute une colonne au jeu de données (ex : score, flag ML...)."""
        self.dataframe[nom] = valeurs

    def supprimerColonne(self, nom: str) -> None:
        if nom in self.dataframe.columns:
            self.dataframe = self.dataframe.drop(columns=[nom])

    def ajouterLignes(self, df: pd.DataFrame) -> None:
        """Ajoute des lignes saisies manuellement au jeu de données existant.

        Générique : si le DataFrame saisi contient des colonnes absentes du
        jeu existant, elles sont créées (remplies de NaN pour les anciennes
        lignes) ; si des colonnes existantes ne sont pas renseignées dans la
        saisie, elles sont simplement laissées à NaN pour les nouvelles
        lignes. L'index est réinitialisé pour rester cohérent avec
        getNbLignes()/getNbColonnes()."""
        self.dataframe = pd.concat(
            [self.dataframe, df], ignore_index=True, sort=False
        )

    def getColonnes(self) -> list[str]:
        return self.colonnes

    def getNbLignes(self) -> int:
        return self.nombreLignes

    def getNbColonnes(self) -> int:
        return self.nombreColonnes

    def copie(self) -> "JeuDonnees":
        return JeuDonnees(
            idDataset=self.idDataset,
            dataframe=self.dataframe.copy(),
            dateImport=self.dateImport,
        )
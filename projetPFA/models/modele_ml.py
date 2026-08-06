"""
models/modele_ml.py
---------------------
Bloc 4 - MACHINE LEARNING : détection des lignes "non utiles" dans un
fichier de données quelconque (pas de colonnes connues à l'avance).

Comme demandé, deux étapes :
1. Génération automatique d'une cible binaire (0 = Non_utile, 1 = Utile)
   à partir de règles génériques appliquées à N'IMPORTE QUEL DataFrame :
   ligne trop incomplète, ligne dupliquée, valeur numérique aberrante.
   (voir `generer_labels_utilite`)
2. Entraînement d'un classifieur supervisé sur cette cible, afin que le
   modèle apprenne à généraliser ces règles à partir des autres
   caractéristiques de la ligne — utile notamment quand une ligne
   "limite" (ex. 40% de valeurs manquantes, sous le seuil des règles)
   ressemble statistiquement aux lignes rejetées.

- ModeleML : classe abstraite (entrainer, predire, predireProbabilite,
  sauvegarderModele, chargerModele).
- RandomForestClassifierModel (+ variantes RegressionLogistique,
  GradientBoosting, SVM) : implémentations concrètes via scikit-learn.
"""

from __future__ import annotations

import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from config import (
    FACTEUR_IQR_OUTLIER,
    RF_NB_ARBRES_DEFAUT,
    RF_PROFONDEUR_MAX_DEFAUT,
    RF_RANDOM_STATE_DEFAUT,
    SEUIL_TAUX_MANQUANT,
)
from models.logger import get_logger
from models.pretraitement import dedupliquer_colonnes

logger = get_logger(__name__)

# Ordre fixe des classes d'utilité -> encodage entier stable (0/1),
# nécessaire car sklearn attend une cible numérique.
CLASSES_UTILITE: list[str] = ["Non_utile", "Utile"]
_UTILITE_VERS_INT = {nom: i for i, nom in enumerate(CLASSES_UTILITE)}
_INT_VERS_UTILITE = {i: nom for nom, i in _UTILITE_VERS_INT.items()}


# --------------------------------------------------------------------------
# Génération générique des labels "utile / non utile"
# --------------------------------------------------------------------------
def generer_labels_utilite(
    df: pd.DataFrame,
    seuil_manquant: float = SEUIL_TAUX_MANQUANT,
    facteur_iqr: float = FACTEUR_IQR_OUTLIER,
) -> tuple[pd.Series, dict]:
    """
    Construit la colonne cible "Utile" (0/1) à partir de règles génériques,
    applicables à N'IMPORTE QUEL fichier (aucun nom de colonne supposé) :

      - trop_vide        : la ligne a plus de `seuil_manquant` % de cellules vides
      - doublon          : la ligne est un doublon exact d'une ligne précédente
      - valeur_aberrante : au moins une colonne numérique contient, sur cette
                            ligne, une valeur hors de [Q1 - k*IQR, Q3 + k*IQR]
                            (méthode IQR, k = `facteur_iqr`)

    Une ligne est étiquetée "Non_utile" (0) si AU MOINS une règle se
    déclenche, sinon "Utile" (1).

    Retourne (labels, rapport) où `labels` est une Series d'entiers 0/1
    alignée sur l'index de `df`, et `rapport` un résumé (nombre de lignes
    concernées par chaque règle) affichable dans l'UI.
    """
    raisons = pd.DataFrame(index=df.index)

    # Règle 1 : ligne trop incomplète
    raisons["trop_vide"] = df.isna().mean(axis=1) > seuil_manquant

    # Règle 2 : ligne strictement dupliquée (on garde la 1ère occurrence)
    raisons["doublon"] = df.duplicated(keep="first")

    # Règle 3 : valeur aberrante sur au moins une colonne numérique (IQR).
    # On retient aussi QUELLES colonnes ont déclenché la règle
    # (`colonnes_aberrantes`) : le pipeline les exclut ensuite des features
    # d'entraînement (cf. PipelinePrincipal.pretraiterDonnees). Sans cela,
    # la valeur brute qui a servi à ÉTIQUETER la ligne resterait aussi dans
    # les features qui servent à la PRÉDIRE : le modèle n'a alors qu'à
    # reseuiller cette même colonne pour reproduire la règle à 100%, au lieu
    # d'apprendre à généraliser à partir des autres caractéristiques de la
    # ligne (ce qui donne des métriques artificiellement parfaites et une
    # importance des variables écrasée sur cette seule colonne).
    colonnes_num = df.select_dtypes(include="number").columns
    aberrant = pd.Series(False, index=df.index)
    colonnes_aberrantes: list[str] = []
    for col in colonnes_num:
        serie = df[col]
        q1, q3 = serie.quantile(0.25), serie.quantile(0.75)
        iqr = q3 - q1
        if pd.isna(iqr) or iqr == 0:
            continue
        borne_basse = q1 - facteur_iqr * iqr
        borne_haute = q3 + facteur_iqr * iqr
        masque = (serie < borne_basse) | (serie > borne_haute)
        if masque.any():
            colonnes_aberrantes.append(col)
        aberrant = aberrant | masque
    raisons["valeur_aberrante"] = aberrant

    non_utile = raisons.any(axis=1)
    labels = (~non_utile).astype(int)  # 1 = Utile, 0 = Non_utile

    rapport = {
        "trop_vide": int(raisons["trop_vide"].sum()),
        "doublon": int(raisons["doublon"].sum()),
        "valeur_aberrante": int(raisons["valeur_aberrante"].sum()),
        "non_utile_total": int(non_utile.sum()),
        "utile_total": int((~non_utile).sum()),
        "colonnes_aberrantes": colonnes_aberrantes,
    }
    logger.info("Labels d'utilité générés (règles génériques) : %s", rapport)
    return labels, rapport


def utilite_vers_texte(labels: pd.Series | np.ndarray) -> pd.Series:
    """Convertit des labels entiers (0/1) en libellés lisibles (Non_utile/Utile)."""
    return pd.Series(labels).map(_INT_VERS_UTILITE)


def construire_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construit un jeu de caractéristiques numériques exploitable par les
    modèles scikit-learn à partir du DataFrame prétraité (déjà encodé).

    `pd.get_dummies` (utilisé par `encoderVariables`) produit des colonnes
    de type `bool` sur les versions récentes de pandas : il faut donc les
    inclure explicitement en plus de `np.number`, sinon toutes les
    variables catégorielles encodées sont silencieusement exclues.
    """
    features = df.select_dtypes(include=[np.number, "bool"]).copy()
    features = dedupliquer_colonnes(features)
    features = features.astype(float)
    features = features.fillna(0)
    return features


# --------------------------------------------------------------------------
# Classe abstraite ModeleML
# --------------------------------------------------------------------------
class ModeleML(ABC):
    """Classe abstraite commune à tous les modèles de Machine Learning."""

    def __init__(self) -> None:
        self.modele = None

    @abstractmethod
    def entrainer(self, X: pd.DataFrame, y: pd.Series) -> None:
        ...

    @abstractmethod
    def predire(self, X: pd.DataFrame) -> np.ndarray:
        ...

    @abstractmethod
    def predireProbabilite(self, X: pd.DataFrame) -> np.ndarray:
        ...

    def sauvegarderModele(self, chemin: str) -> None:
        with open(chemin, "wb") as f:
            pickle.dump(self.modele, f)
        logger.info("Modèle sauvegardé : %s", chemin)

    def chargerModele(self, chemin: str) -> None:
        with open(chemin, "rb") as f:
            self.modele = pickle.load(f)
        logger.info("Modèle chargé : %s", chemin)


# --------------------------------------------------------------------------
# Implémentation concrète : RandomForestClassifier
# --------------------------------------------------------------------------
@dataclass
class RandomForestClassifierModel(ModeleML):
    nombreArbres: int = RF_NB_ARBRES_DEFAUT
    profondeurMax: Optional[int] = RF_PROFONDEUR_MAX_DEFAUT
    class_weight: str | None = "balanced"
    randomState: int = RF_RANDOM_STATE_DEFAUT
    _feature_names: list[str] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        ModeleML.__init__(self)
        self.modele = RandomForestClassifier(
            n_estimators=self.nombreArbres,
            max_depth=self.profondeurMax,
            class_weight=self.class_weight,
            random_state=self.randomState,
        )

    def entrainer(self, X: pd.DataFrame, y: pd.Series) -> None:
        self._feature_names = list(X.columns)
        self.modele.fit(X, y)
        logger.info(
            "RandomForestClassifier entraîné : %d arbres, %d features",
            self.nombreArbres, len(self._feature_names),
        )

    def predire(self, X: pd.DataFrame) -> np.ndarray:
        X = X.reindex(columns=self._feature_names, fill_value=0)
        return self.modele.predict(X)

    def predireProbabilite(self, X: pd.DataFrame) -> np.ndarray:
        X = X.reindex(columns=self._feature_names, fill_value=0)
        return self.modele.predict_proba(X)

    def importance_variables(self) -> pd.Series:
        return pd.Series(
            self.modele.feature_importances_, index=self._feature_names
        ).sort_values(ascending=False)


@dataclass
class RegressionLogistiqueModel(ModeleML):
    C: float = 1.0
    maxIter: int = 500
    class_weight: str | None = "balanced"
    randomState: int = RF_RANDOM_STATE_DEFAUT
    _feature_names: list[str] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        ModeleML.__init__(self)
        self.modele = LogisticRegression(
            C=self.C,
            max_iter=self.maxIter,
            class_weight=self.class_weight,
            random_state=self.randomState,
        )

    def entrainer(self, X: pd.DataFrame, y: pd.Series) -> None:
        self._feature_names = list(X.columns)
        self.modele.fit(X, y)
        logger.info("LogisticRegression entraîné : %d features", len(self._feature_names))

    def predire(self, X: pd.DataFrame) -> np.ndarray:
        X = X.reindex(columns=self._feature_names, fill_value=0)
        return self.modele.predict(X)

    def predireProbabilite(self, X: pd.DataFrame) -> np.ndarray:
        X = X.reindex(columns=self._feature_names, fill_value=0)
        return self.modele.predict_proba(X)

    def importance_variables(self) -> pd.Series:
        coefs = np.abs(self.modele.coef_[0])
        return pd.Series(coefs, index=self._feature_names).sort_values(ascending=False)


@dataclass
class GradientBoostingModel(ModeleML):
    nombreArbres: int = 150
    tauxApprentissage: float = 0.1
    profondeurMax: int = 3
    randomState: int = RF_RANDOM_STATE_DEFAUT
    _feature_names: list[str] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        ModeleML.__init__(self)
        self.modele = GradientBoostingClassifier(
            n_estimators=self.nombreArbres,
            learning_rate=self.tauxApprentissage,
            max_depth=self.profondeurMax,
            random_state=self.randomState,
        )

    def entrainer(self, X: pd.DataFrame, y: pd.Series) -> None:
        self._feature_names = list(X.columns)
        self.modele.fit(X, y)
        logger.info(
            "GradientBoostingClassifier entraîné : %d arbres, %d features",
            self.nombreArbres, len(self._feature_names),
        )

    def predire(self, X: pd.DataFrame) -> np.ndarray:
        X = X.reindex(columns=self._feature_names, fill_value=0)
        return self.modele.predict(X)

    def predireProbabilite(self, X: pd.DataFrame) -> np.ndarray:
        X = X.reindex(columns=self._feature_names, fill_value=0)
        return self.modele.predict_proba(X)

    def importance_variables(self) -> pd.Series:
        return pd.Series(
            self.modele.feature_importances_, index=self._feature_names
        ).sort_values(ascending=False)


@dataclass
class SVMModel(ModeleML):
    C: float = 1.0
    kernel: str = "rbf"
    class_weight: str | None = "balanced"
    randomState: int = RF_RANDOM_STATE_DEFAUT
    _feature_names: list[str] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        ModeleML.__init__(self)
        self.modele = SVC(
            C=self.C,
            kernel=self.kernel,
            probability=True,
            class_weight=self.class_weight,
            random_state=self.randomState,
        )

    def entrainer(self, X: pd.DataFrame, y: pd.Series) -> None:
        self._feature_names = list(X.columns)
        self.modele.fit(X, y)
        logger.info("SVC entraîné : noyau=%s, %d features", self.kernel, len(self._feature_names))

    def predire(self, X: pd.DataFrame) -> np.ndarray:
        X = X.reindex(columns=self._feature_names, fill_value=0)
        return self.modele.predict(X)

    def predireProbabilite(self, X: pd.DataFrame) -> np.ndarray:
        X = X.reindex(columns=self._feature_names, fill_value=0)
        return self.modele.predict_proba(X)

    # Pas d'importance_variables() : pas d'équivalent direct pour un noyau
    # non-linéaire (rbf/poly). L'UI (app.py) gère déjà ce cas avec hasattr().

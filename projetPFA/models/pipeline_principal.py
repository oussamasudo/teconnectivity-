"""
models/pipeline_principal.py
-------------------------------
Classe PipelinePrincipal : orchestre l'ensemble du pipeline, sur
N'IMPORTE QUEL fichier Excel/CSV (colonnes inconnues à l'avance) :

  1. Import du fichier            -> FichierExcel -> JeuDonnees
  2. Analyse exploratoire (EDA)    -> AnalyseEDA
  3. Prétraitement                 -> Pretraitement
  4. Machine Learning              -> détection des lignes non utiles
  5. Export du résultat            -> ExportExcel
  6. Restitution (hors code)       -> Power BI (externe)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split

from config import COLONNE_CIBLE, COLONNES_OBLIGATOIRES, TAILLE_TEST_DEFAUT
from models.analyse_eda import AnalyseEDA
from models.export_excel import ExportExcel
from models.fichier_excel import FichierExcel
from models.jeu_donnees import JeuDonnees
from models.logger import get_logger
from models.modele_ml import (
    CLASSES_UTILITE,
    GradientBoostingModel,
    ModeleML,
    RandomForestClassifierModel,
    RegressionLogistiqueModel,
    SVMModel,
    construire_features,
    generer_labels_utilite,
    utilite_vers_texte,
)
from models.pretraitement import Pretraitement
from models.rapport_evaluation import RapportEvaluation

logger = get_logger(__name__)

# Registre des modèles disponibles pour l'étape 4 (Machine Learning).
# Clé = libellé affiché dans le selectbox Streamlit -> classe du modèle.
MODELES_DISPONIBLES: dict[str, type[ModeleML]] = {
    "Random Forest": RandomForestClassifierModel,
    "Régression Logistique": RegressionLogistiqueModel,
    "Gradient Boosting": GradientBoostingModel,
    "SVM (Support Vector Machine)": SVMModel,
}


def detecter_colonnes_texte_libre(
    df: pd.DataFrame,
    seuil_cardinalite: float = 0.5,
    longueur_moyenne_max: int = 30,
) -> list[str]:
    """Détecte GENERIQUEMENT les colonnes de "texte libre" (commentaires,
    descriptions, identifiants quasi uniques...) à exclure des features ML
    car un encodage catégoriel classique (One-Hot) les ferait exploser en
    des centaines de colonnes sans valeur prédictive. Ne dépend d'aucun nom
    de colonne particulier : uniquement du contenu (cardinalité, longueur).
    """
    colonnes: list[str] = []
    n = len(df)
    if n == 0:
        return colonnes
    for col in df.select_dtypes(include=["object", "category"]).columns:
        serie = df[col].dropna().astype(str)
        if serie.empty:
            continue
        cardinalite = serie.nunique() / n
        longueur_moyenne = serie.str.len().mean()
        if cardinalite > seuil_cardinalite or longueur_moyenne > longueur_moyenne_max:
            colonnes.append(col)
    return colonnes


@dataclass
class PipelinePrincipal:
    fichierExcel: Optional[FichierExcel] = None
    jeuDonnees: Optional[JeuDonnees] = None
    pretraitement: Pretraitement = field(default_factory=Pretraitement)
    analyseEDA: AnalyseEDA = field(default_factory=AnalyseEDA)
    modele: Optional[ModeleML] = None
    rapport: Optional[RapportEvaluation] = None
    exportExcel: ExportExcel = field(default_factory=ExportExcel)

    def __post_init__(self) -> None:
        if self.exportExcel is None:
            self.exportExcel = ExportExcel()

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        if getattr(self, "exportExcel", None) is None:
            self.exportExcel = ExportExcel()

    # état interne conservé entre les étapes
    _jeu_brut: Optional[JeuDonnees] = field(default=None, repr=False)
    _jeu_nettoye: Optional[JeuDonnees] = field(default=None, repr=False)    # dédupliqué + imputé, LISIBLE
    _jeu_pretraite: Optional[JeuDonnees] = field(default=None, repr=False)  # + scaling + encodage, pour le ML
    _labels: Optional[pd.Series] = field(default=None, repr=False)
    _colonnes_texte_libre: list[str] = field(default_factory=list, repr=False)
    rapport_regles: dict = field(default_factory=dict, repr=False)
    source_cible: str = field(default="", repr=False)

    # "fichier" (import Excel/CSV) ou "manuel" (saisie manuelle dans l'UI).
    # Sert à masquer, côté app.py, les valeurs manquantes et l'importance
    # des variables lorsque les données viennent d'une saisie manuelle
    # (peu de lignes, peu représentatif -> ces indicateurs n'ont pas de sens).
    source_import: str = field(default="fichier", repr=False)

    def estSaisieManuelle(self) -> bool:
        """À utiliser côté UI pour conditionner l'affichage : True si les
        données actuelles proviennent d'une saisie manuelle plutôt que
        d'un fichier importé."""
        return self.source_import == "manuel"

    # ------------------------------------------------------------------
    # 1. Import du fichier
    # ------------------------------------------------------------------
    def importerDonnees(self, fichierExcel: FichierExcel) -> JeuDonnees:
        self.fichierExcel = fichierExcel
        fichierExcel.verifierFormat()
        fichierExcel.charger()
        fichierExcel.verifierColonnes(COLONNES_OBLIGATOIRES)
        self.jeuDonnees = fichierExcel.creerJeuDonnees()
        self._jeu_brut = self.jeuDonnees
        self.source_import = "fichier"
        return self.jeuDonnees

    def importerDonneesManuelles(
        self, dataframe: pd.DataFrame, nom: str = "Saisie manuelle"
    ) -> JeuDonnees:
        """Crée le jeu de données directement à partir d'un DataFrame saisi
        à la main dans l'interface (page Import, onglet « Saisie manuelle »),
        sans passer par un fichier physique ni par FichierExcel."""
        self.fichierExcel = None
        jeu = JeuDonnees(idDataset=1, dataframe=dataframe.reset_index(drop=True).copy())
        self.jeuDonnees = jeu
        self._jeu_brut = jeu
        self.source_import = "manuel"
        logger.info(
            "Jeu de données créé depuis une saisie manuelle (%s) : %d lignes, %d colonnes",
            nom, jeu.getNbLignes(), jeu.getNbColonnes(),
        )
        return jeu

    def ajouterLignesManuelles(self, dataframe: pd.DataFrame) -> JeuDonnees:
        """Ajoute des lignes saisies manuellement au jeu de données déjà
        importé (utilisé depuis la page EDA, pour compléter/corriger après
        coup). Si aucun jeu de données n'existe encore, se comporte comme
        importerDonneesManuelles()."""
        if self.jeuDonnees is None:
            return self.importerDonneesManuelles(dataframe)
        self.jeuDonnees.ajouterLignes(dataframe)
        self._jeu_brut = self.jeuDonnees
        logger.info(
            "%d ligne(s) manuelle(s) ajoutée(s) : %d lignes au total",
            len(dataframe), self.jeuDonnees.getNbLignes(),
        )
        return self.jeuDonnees

    # ------------------------------------------------------------------
    # 2. Analyse exploratoire (EDA)
    # ------------------------------------------------------------------
    def analyserDonnees(self) -> AnalyseEDA:
        if self.jeuDonnees is None:
            raise RuntimeError("importerDonnees() doit être appelé avant analyserDonnees().")
        self.analyseEDA.calculerStatistiques(self.jeuDonnees)
        return self.analyseEDA

    # ------------------------------------------------------------------
    # 3. Prétraitement + génération de la cible utile/non-utile
    # ------------------------------------------------------------------
    def pretraiterDonnees(self) -> JeuDonnees:
        if self.jeuDonnees is None:
            raise RuntimeError("importerDonnees() doit être appelé avant pretraiterDonnees().")

        # 1) Génère (ou récupère) les labels utile/non-utile AVANT tout
        #    nettoyage, sur l'index d'origine complet (nécessaire pour
        #    pouvoir détecter correctement la règle "doublon").
        df_brut = self.jeuDonnees.dataframe
        if COLONNE_CIBLE in df_brut.columns:
            # Le fichier fournit déjà une cible ("Utile"/"Non_utile" ou 0/1)
            # -> on l'utilise comme vérité terrain plutôt que les règles.
            col = df_brut[COLONNE_CIBLE]
            if col.dtype == object:
                labels_brutes = col.map({c: i for i, c in enumerate(CLASSES_UTILITE)})
            else:
                labels_brutes = col.astype(int)
            self.rapport_regles = {}
            self.source_cible = "colonne_existante"
        else:
            labels_brutes, self.rapport_regles = generer_labels_utilite(df_brut)
            self.source_cible = "regles_generiques"

        jeu_a_traiter = self.jeuDonnees.copie()
        jeu_a_traiter.supprimerColonne(COLONNE_CIBLE)

        # 2) Nettoyage "lisible" : suppression des doublons, extraction des
        #    features de date, imputation des NaN.
        #    L'index d'origine est conservé -> permet de réaligner les labels.
        jeu_dedup = self.pretraitement.supprimerDoublons(jeu_a_traiter)
        jeu_dates = self.pretraitement.extraireFeaturesDate(jeu_dedup)
        self._jeu_nettoye = self.pretraitement.traiterValeursManquantes(jeu_dates)

        # 3) Réalignement des labels sur les lignes restantes après dédoublonnage.
        self._labels = labels_brutes.loc[self._jeu_nettoye.dataframe.index]

        # 4) Transformation ML (normalisation/standardisation + encodage),
        #    calculée à partir du jeu nettoyé (mêmes index). Les colonnes de
        #    texte libre (détectées génériquement, cf. plus haut) en sont
        #    exclues : elles restent dans le jeu "lisible" pour l'export.
        jeu_features = self._jeu_nettoye.copie()
        self._colonnes_texte_libre = detecter_colonnes_texte_libre(jeu_features.dataframe)
        for col in self._colonnes_texte_libre:
            jeu_features.supprimerColonne(col)

        colonnes_num = list(jeu_features.dataframe.select_dtypes(include="number").columns)
        if self.pretraitement.methodeNormalisation == "minmax":
            jeu_features = self.pretraitement.normaliser(jeu_features, colonnes_num)
        else:
            jeu_features = self.pretraitement.standardiser(jeu_features, colonnes_num)

        colonnes_cat = list(
            jeu_features.dataframe.select_dtypes(include=["object", "category"]).columns
        )
        jeu_features = self.pretraitement.encoderVariables(jeu_features, colonnes_cat)

        self._jeu_pretraite = jeu_features
        return self._jeu_pretraite

    # ------------------------------------------------------------------
    # 4. Machine Learning (Random Forest / Régression Logistique /
    #    Gradient Boosting / SVM, au choix via nomModele)
    # ------------------------------------------------------------------
    def entrainerModele(
        self,
        nomModele: str = "Random Forest",
        parametres: Optional[dict] = None,
        randomState: int = 42,
        taille_test: float = TAILLE_TEST_DEFAUT,
    ) -> RapportEvaluation:
        if self._jeu_pretraite is None:
            raise RuntimeError("pretraiterDonnees() doit être appelé avant entrainerModele().")
        if nomModele not in MODELES_DISPONIBLES:
            raise ValueError(
                f"Modèle inconnu : {nomModele!r}. Choix possibles : {list(MODELES_DISPONIBLES)}"
            )

        X = construire_features(self._jeu_pretraite.dataframe)
        y = self._labels.reset_index(drop=True)
        X = X.reset_index(drop=True)

        if y.nunique() < 2:
            seule_classe = utilite_vers_texte(pd.Series([y.iloc[0]])).iloc[0]
            raise ValueError(
                "Impossible d'entraîner un modèle : la cible ne contient qu'une seule "
                f"classe ({seule_classe!r}) sur l'ensemble du fichier. Il faut au moins "
                "quelques lignes de chaque catégorie (Utile ET Non_utile) pour qu'un "
                "classifieur ait quelque chose à apprendre."
            )

        peut_stratifier = y.value_counts().min() >= 2
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=taille_test, random_state=randomState,
            stratify=y if peut_stratifier else None,
        )

        # Filet de sécurité générique : quelle que soit la méthode de split,
        # un jeu d'ENTRAINEMENT réduit à une seule classe rend l'ajustement
        # du modèle impossible (ex. classe minoritaire ramenée à un seul
        # exemple sur tout le fichier -> pas assez pour stratifier -> un
        # split aléatoire peut, par hasard, l'envoyer entièrement côté
        # test). On rebascule alors manuellement vers le train un exemple
        # de chaque classe manquante, plutôt que de laisser sklearn planter.
        classes_manquantes = set(y.unique()) - set(y_train.unique())
        for classe in classes_manquantes:
            idx_a_deplacer = y_test[y_test == classe].index[:1]
            X_train = pd.concat([X_train, X_test.loc[idx_a_deplacer]])
            y_train = pd.concat([y_train, y_test.loc[idx_a_deplacer]])
            X_test = X_test.drop(index=idx_a_deplacer)
            y_test = y_test.drop(index=idx_a_deplacer)

        classe_modele = MODELES_DISPONIBLES[nomModele]
        self.modele = classe_modele(randomState=randomState, **(parametres or {}))
        self.modele.entrainer(X_train, y_train)

        y_pred_test = self.modele.predire(X_test)
        y_score = None
        try:
            y_score = self.modele.predireProbabilite(X_test)[:, 1]
        except Exception:
            y_score = None

        self.rapport = RapportEvaluation().calculerMetriques(
            y_test,
            y_pred_test,
            labels=list(range(len(CLASSES_UTILITE))),
            y_score=y_score,
        )

        # Validation croisée si l'échantillon est suffisant.
        try:
            min_labels = y.value_counts().min()
            n_splits = min(5, min_labels) if min_labels >= 2 else 0
            if n_splits >= 2 and len(X) >= 10:
                cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=randomState)
                scoring = [
                    "accuracy",
                    "precision_weighted",
                    "recall_weighted",
                    "f1_weighted",
                    "roc_auc",
                ]
                results = cross_validate(
                    self.modele,
                    X,
                    y,
                    cv=cv,
                    scoring=scoring,
                    n_jobs=-1,
                    error_score="raise",
                )
                self.rapport.crossValScores = {
                    metric: float(results[f"test_{metric}"].mean())
                    for metric in scoring
                    if f"test_{metric}" in results
                }
        except Exception:
            self.rapport.crossValScores = {}

        return self.rapport

    def predire(self) -> pd.DataFrame:
        """Prédit, pour CHAQUE ligne du jeu nettoyé (dédupliqué), si elle est
        "Utile" ou "Non_utile" + le score de confiance associé (probabilité
        de la classe prédite). Le résultat reprend les valeurs d'origine
        LISIBLES (jeu_nettoye), pas les valeurs transformées pour le ML."""
        if self.modele is None:
            raise RuntimeError("entrainerModele() doit être appelé avant predire().")

        X = construire_features(self._jeu_pretraite.dataframe)
        predictions = self.modele.predire(X)
        probabilites = self.modele.predireProbabilite(X)
        score_confiance = probabilites.max(axis=1)

        # jeu_nettoye et jeu_pretraite (donc X) partagent le même index.
        resultat = self._jeu_nettoye.dataframe.copy()
        resultat["utilite_predite"] = utilite_vers_texte(
            pd.Series(predictions, index=X.index)
        ).values
        resultat["score_confiance"] = pd.Series(score_confiance, index=X.index)
        return resultat.reset_index(drop=True)

    # ------------------------------------------------------------------
    # 5. Export du résultat
    # ------------------------------------------------------------------
    def exporterResultat(self, df_resultat: pd.DataFrame) -> dict[str, pd.DataFrame]:
        """Répartit les lignes par classe prédite (Utile / Non_utile)."""
        return {
            classe: df_resultat[df_resultat["utilite_predite"] == classe]
            for classe in CLASSES_UTILITE
        }

    # ------------------------------------------------------------------
    # Pipeline complet (utilisé hors Streamlit, ex. script CLI)
    # ------------------------------------------------------------------
    def executerPipeline(self, fichierExcel: FichierExcel) -> dict:
        self.importerDonnees(fichierExcel)
        self.analyserDonnees()
        self.pretraiterDonnees()
        rapport = self.entrainerModele()
        df_resultat = self.predire()
        par_classe = self.exporterResultat(df_resultat)

        chemin_final = self.exportExcel.enregistrer(
            df_resultat, nom_fichier="Donnees_Nettoyees.xlsx"
        )

        return {
            "rapport_evaluation": rapport.to_dict(),
            **{f"nb_{classe.lower()}": len(df) for classe, df in par_classe.items()},
            "fichier_final": chemin_final,
        }
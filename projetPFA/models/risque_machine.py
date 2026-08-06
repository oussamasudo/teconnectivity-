"""
models/risque_machine.py
--------------------------
Prédiction de la prochaine panne par machine (maintenance prédictive) —
module additionnel, indépendant du pipeline ML de détection de lignes
utiles/non utiles (models/pipeline_principal.py, models/modele_ml.py).

Remplace l'ancien classement par score de règles (fréquence/durée/tendance
pondérées) — redondant avec les Top N / Pareto déjà proposés par l'EDA —
par un VRAI modèle supervisé, entraîné sur l'historique des pannes de
TOUTES les machines confondues (un seul modèle global : le peu de pannes
par machine, pris isolément, ne permettrait pas d'entraîner un modèle
fiable machine par machine) :

  1. Reconstruction de l'historique de chaque machine : dates de pannes
     triées, calcul des intervalles entre pannes consécutives.
  2. Chaque intervalle observé (panne i-1 -> panne i) devient un EXEMPLE
     d'entraînement : les features sont calculées uniquement à partir de
     l'historique connu JUSQU'A la panne i-1 (pas de fuite), la cible est
     le nombre de jours réellement écoulé jusqu'à la panne i.
  3. Deux modèles sont entraînés sur ces exemples :
       - un classifieur : probabilité qu'une panne survienne dans les
         `seuilJoursRisque` prochains jours ("panne imminente") ;
       - un régresseur : délai estimé (en jours) avant la prochaine panne,
         calculé et affiché pour TOUTES les machines du fichier (c'est le
         résultat central de ce module), quel que soit leur niveau de
         risque.
  4. Application aux machines du fichier : les mêmes features sont
     recalculées à partir de la DERNIÈRE panne connue de chaque machine,
     puis passées aux deux modèles pour obtenir la prédiction "vers l'avant".

Générique : ne dépend d'aucun nom de colonne fixe. Les colonnes
"machine" / "date" / "durée d'arrêt" sont fournies par l'appelant (l'UI
les propose via colonnes_machine_candidates / colonnes_date_candidates).
La colonne date est ici OBLIGATOIRE (contrairement à l'ancien module) :
prédire un délai avant la prochaine panne n'a pas de sens sans historique
temporel.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error, roc_auc_score
from sklearn.model_selection import train_test_split

from config import RF_NB_ARBRES_DEFAUT
from models.logger import get_logger

logger = get_logger(__name__)

# Charte graphique TE Connectivity (cohérente avec les autres modules)
TE_ORANGE = "#FF8200"
TE_ANTHRACITE = "#2B2B2B"
TE_GRIS = "#6A6A6A"

NIVEAUX_RISQUE = ["Faible", "Modéré", "Élevé", "Critique"]
COULEURS_NIVEAU = {
    "Faible": "#8FBF7F",
    "Modéré": "#FFD34D",
    "Élevé": "#FF9A33",
    "Critique": "#C0392B",
}

SEUIL_JOURS_RISQUE_DEFAUT = 30    # "panne imminente" = dans les 30 prochains jours
FENETRE_TENDANCE_JOURS_DEFAUT = 30
MIN_EXEMPLES_ENTRAINEMENT = 10    # en dessous, pas assez d'historique pour un modèle fiable
PERCENTILE_SEUIL_CRITIQUE_DEFAUT = 0.95  # calibré sur la distribution du fichier, pas une constante fixe


def _normaliser_0_1(serie: pd.Series) -> pd.Series:
    """Normalisation min-max ; retourne 0 partout si la série est
    constante ou vide (évite une division par zéro)."""
    if serie.empty:
        return serie
    mini, maxi = serie.min(), serie.max()
    if pd.isna(mini) or pd.isna(maxi) or maxi == mini:
        return pd.Series(0.0, index=serie.index)
    return (serie - mini) / (maxi - mini)


@dataclass
class PredicteurPanneMachine:
    """Prédit, pour chaque machine, la probabilité d'une panne imminente et
    le délai estimé avant celle-ci, à partir d'un modèle global entraîné sur
    l'historique des intervalles entre pannes de toutes les machines."""

    seuilJoursRisque: int = SEUIL_JOURS_RISQUE_DEFAUT
    fenetreTendanceJours: int = FENETRE_TENDANCE_JOURS_DEFAUT
    randomState: int = 42

    _feature_names: list[str] = field(default_factory=list, repr=False)
    _medianes_imputation: dict[str, float] = field(default_factory=dict, repr=False)
    _classifieur: Optional[RandomForestClassifier] = field(default=None, repr=False)
    _regresseur: Optional[RandomForestRegressor] = field(default=None, repr=False)
    metriques: dict = field(default_factory=dict, repr=False)

    # ------------------------------------------------------------------
    # Détection générique des colonnes exploitables (proposées à l'UI)
    # ------------------------------------------------------------------
    def colonnes_machine_candidates(self, df: pd.DataFrame) -> list[str]:
        """Suggère les colonnes catégorielles susceptibles de représenter
        un équipement/une machine : d'abord celles dont le nom contient un
        mot-clé pertinent, puis les autres colonnes catégorielles non
        constantes (fallback générique, aucun nom de colonne supposé)."""
        mots_cles = [
            "machine", "équipement", "equipement", "asset",
            "ligne", "line", "equipment", "poste",
        ]
        colonnes_cat = df.select_dtypes(include=["object", "category"]).columns
        prioritaires = [c for c in colonnes_cat if any(m in c.lower() for m in mots_cles)]
        autres = [
            c for c in colonnes_cat
            if c not in prioritaires and df[c].nunique(dropna=True) >= 2
        ]
        return prioritaires + autres

    def colonnes_date_candidates(self, df: pd.DataFrame) -> list[str]:
        """Suggère les colonnes susceptibles de contenir une date (déjà
        typées datetime, ou texte majoritairement parsable en date)."""
        candidates: list[str] = []
        for col in df.columns:
            serie = df[col]
            if pd.api.types.is_datetime64_any_dtype(serie):
                candidates.append(col)
                continue
            # `serie.dtype == object` ne détecte plus les colonnes texte sur
            # pandas >= 3 (nouveau dtype "str" dédié, différent d'`object`) :
            # on se base sur is_numeric/is_datetime plutôt que sur `object`
            # pour rester compatible avec les deux versions de pandas.
            est_numerique = pd.api.types.is_numeric_dtype(serie)
            if not est_numerique:
                echantillon = serie.dropna().astype(str).head(30)
                if echantillon.empty:
                    continue
                parsed = pd.to_datetime(echantillon, errors="coerce")
                if parsed.notna().mean() >= 0.7:
                    candidates.append(col)
        candidates.sort(key=lambda c: 0 if "date" in c.lower() else 1)
        return candidates

    def colonnes_compteur_usage_candidates(self, df: pd.DataFrame) -> list[str]:
        """Suggère les colonnes numériques susceptibles de représenter un
        compteur d'usage cumulé (heures de fonctionnement, cycles,
        production...), utilisées comme "horloge" de substitution quand le
        fichier ne fournit ni date ni historique de pannes répétées (ex. un
        instantané "1 ligne = 1 machine")."""
        mots_cles = [
            "heure", "hour", "fonctionnement", "usage", "cycle",
            "compteur", "runtime", "production", "age", "âge", "km",
        ]
        colonnes_num = [
            c for c in df.select_dtypes(include=[np.number]).columns if df[c].notna().any()
        ]
        prioritaires = [c for c in colonnes_num if any(m in c.lower() for m in mots_cles)]
        autres = [c for c in colonnes_num if c not in prioritaires]
        return prioritaires + autres

    def colonnes_taux_risque_candidates(self, df: pd.DataFrame) -> list[str]:
        """Suggère les colonnes numériques susceptibles de représenter un
        taux de défaut/risque (pourcentage ou proportion) : nom évocateur en
        priorité, puis colonnes dont les valeurs restent dans un intervalle
        plausible de pourcentage (0-100)."""
        mots_cles = [
            "taux", "defaut", "défaut", "defect", "panne", "risque",
            "echec", "échec", "failure", "rate", "%",
        ]
        colonnes_num = [
            c for c in df.select_dtypes(include=[np.number]).columns if df[c].notna().any()
        ]
        prioritaires = [c for c in colonnes_num if any(m in c.lower() for m in mots_cles)]
        plage_pourcentage = [
            c for c in colonnes_num
            if c not in prioritaires and df[c].dropna().between(0, 100).mean() >= 0.95
        ]
        return prioritaires + plage_pourcentage

    # ------------------------------------------------------------------
    # Préparation de l'historique (1 ligne = 1 panne, trié par machine/date)
    # ------------------------------------------------------------------
    def _preparer_historique(
        self,
        df: pd.DataFrame,
        colonne_machine: str,
        colonne_date: str,
        colonne_duree: Optional[str],
    ) -> pd.DataFrame:
        if colonne_machine not in df.columns or colonne_date not in df.columns:
            return pd.DataFrame()

        travail = pd.DataFrame({
            "machine": df[colonne_machine].astype(str).str.strip(),
            "date": pd.to_datetime(df[colonne_date], errors="coerce"),
        })
        if colonne_duree and colonne_duree in df.columns:
            travail["duree"] = pd.to_numeric(df[colonne_duree], errors="coerce")
        else:
            travail["duree"] = np.nan

        travail = travail[
            travail["machine"].notna()
            & (travail["machine"] != "")
            & (travail["machine"].str.lower() != "nan")
        ]
        travail = travail.dropna(subset=["date"])
        return travail.sort_values(["machine", "date"]).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Features connues juste après la `indice`-ième panne d'une machine
    # (indices 0..indice-1 de `dates`/`durees`), sans aucune information
    # postérieure : utilisé à la fois pour construire un exemple
    # d'entraînement (indice < n, cible = panne suivante réelle) et pour
    # l'état actuel d'une machine (indice = n, prédiction vers l'avant).
    # ------------------------------------------------------------------
    def _features_a_partir_historique(
        self, dates: list[pd.Timestamp], durees: list[float], indice: int
    ) -> dict:
        d_reference = dates[indice - 1]
        gaps = [(dates[j] - dates[j - 1]).days for j in range(1, indice)]
        durees_connues = [d for d in durees[:indice] if pd.notna(d)]
        seuil_recent = d_reference - pd.Timedelta(days=self.fenetreTendanceJours)
        pannes_recentes = sum(1 for d in dates[:indice] if d > seuil_recent)

        return {
            "rang_panne": indice,
            "intervalle_precedent_jours": float(gaps[-1]) if gaps else np.nan,
            "intervalle_moyen_jours": float(np.mean(gaps)) if gaps else np.nan,
            "duree_moyenne_historique": float(np.mean(durees_connues)) if durees_connues else np.nan,
            "duree_derniere": float(durees[indice - 1]) if pd.notna(durees[indice - 1]) else np.nan,
            "pannes_recentes_30j": pannes_recentes,
            "mois_derniere_panne": d_reference.month,
            "jour_semaine_derniere_panne": d_reference.dayofweek,
        }

    def _construire_exemples_entrainement(self, historique: pd.DataFrame) -> pd.DataFrame:
        lignes = []
        for machine, groupe in historique.groupby("machine"):
            dates = groupe["date"].tolist()
            durees = groupe["duree"].tolist()
            n = len(dates)
            for i in range(1, n):
                feats = self._features_a_partir_historique(dates, durees, i)
                feats["machine"] = machine
                feats["delai_jours"] = float((dates[i] - dates[i - 1]).days)
                lignes.append(feats)
        return pd.DataFrame(lignes)

    def _encoder_features(self, df_feats: pd.DataFrame, entrainement: bool) -> pd.DataFrame:
        """Encode les features (one-hot sur "machine") et impute les valeurs
        manquantes. En entraînement, les colonnes et médianes d'imputation
        sont mémorisées pour être réappliquées à l'identique en prédiction
        (cohérence indispensable pour un RandomForest)."""
        df_encode = pd.get_dummies(df_feats, columns=["machine"])
        if entrainement:
            self._medianes_imputation = {
                col: float(df_encode[col].median())
                for col in df_encode.columns
                if pd.api.types.is_numeric_dtype(df_encode[col]) and df_encode[col].isna().any()
            }
            for col, mediane in self._medianes_imputation.items():
                df_encode[col] = df_encode[col].fillna(mediane)
            df_encode = df_encode.fillna(0)
            self._feature_names = list(df_encode.columns)
        else:
            for col, mediane in self._medianes_imputation.items():
                if col in df_encode.columns:
                    df_encode[col] = df_encode[col].fillna(mediane)
            df_encode = df_encode.fillna(0)
            df_encode = df_encode.reindex(columns=self._feature_names, fill_value=0)
        return df_encode.astype(float)

    # ------------------------------------------------------------------
    # Entraînement des deux modèles (classification + régression)
    # ------------------------------------------------------------------
    def entrainer(
        self,
        df: pd.DataFrame,
        colonne_machine: str,
        colonne_date: str,
        colonne_duree: Optional[str] = None,
    ) -> dict:
        historique = self._preparer_historique(df, colonne_machine, colonne_date, colonne_duree)
        exemples = self._construire_exemples_entrainement(historique)

        self._classifieur = None
        self._regresseur = None
        self._feature_names = []
        self._medianes_imputation = {}
        self.metriques = {"nb_exemples_entrainement": len(exemples)}

        if len(exemples) < MIN_EXEMPLES_ENTRAINEMENT:
            return self.metriques

        X = self._encoder_features(exemples.drop(columns=["delai_jours"]), entrainement=True)
        y_delai = exemples["delai_jours"].astype(float)
        y_risque = (y_delai <= self.seuilJoursRisque).astype(int)

        peut_stratifier = y_risque.nunique() == 2 and y_risque.value_counts().min() >= 2
        X_train, X_test, y_delai_train, y_delai_test, y_risque_train, y_risque_test = train_test_split(
            X, y_delai, y_risque, test_size=0.25, random_state=self.randomState,
            stratify=y_risque if peut_stratifier else None,
        )

        self._regresseur = RandomForestRegressor(
            n_estimators=RF_NB_ARBRES_DEFAUT, random_state=self.randomState
        )
        self._regresseur.fit(X_train, y_delai_train)
        y_delai_pred = self._regresseur.predict(X_test)
        self.metriques["mae_jours"] = float(mean_absolute_error(y_delai_test, y_delai_pred))

        if y_risque.nunique() == 2:
            self._classifieur = RandomForestClassifier(
                n_estimators=RF_NB_ARBRES_DEFAUT,
                class_weight="balanced",
                random_state=self.randomState,
            )
            self._classifieur.fit(X_train, y_risque_train)
            y_risque_pred = self._classifieur.predict(X_test)
            self.metriques["accuracy_risque"] = float(accuracy_score(y_risque_test, y_risque_pred))
            try:
                if y_risque_test.nunique() == 2:
                    y_score = self._classifieur.predict_proba(X_test)[:, 1]
                    self.metriques["auc_risque"] = float(roc_auc_score(y_risque_test, y_score))
            except Exception:
                pass

        logger.info(
            "Prédicteur de panne entraîné : %d exemples, métriques=%s",
            len(exemples), self.metriques,
        )
        return self.metriques

    # ------------------------------------------------------------------
    # Prédiction "vers l'avant" pour chaque machine du fichier
    # ------------------------------------------------------------------
    def predire(
        self,
        df: pd.DataFrame,
        colonne_machine: str,
        colonne_date: str,
        colonne_duree: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Retourne un DataFrame (1 ligne par machine), trié par probabilité de
        panne imminente décroissante :
          machine, probabilite_panne_imminente (%), niveau_risque,
          delai_estime_jours (durée prévue avant la prochaine panne, pour
          TOUTES les machines), date_prochaine_panne_estimee,
          nb_pannes_historique, derniere_panne, jours_depuis_derniere_panne.

        Retourne un DataFrame vide si les colonnes fournies ne permettent
        pas de construire d'historique, ou si l'historique disponible est
        trop réduit pour entraîner un modèle fiable
        (< MIN_EXEMPLES_ENTRAINEMENT transitions entre pannes).
        """
        historique = self._preparer_historique(df, colonne_machine, colonne_date, colonne_duree)
        if historique.empty:
            return pd.DataFrame()

        self.entrainer(df, colonne_machine, colonne_date, colonne_duree)
        if self._regresseur is None and self._classifieur is None:
            return pd.DataFrame()

        lignes = []
        for machine, groupe in historique.groupby("machine"):
            dates = groupe["date"].tolist()
            durees = groupe["duree"].tolist()
            feats = self._features_a_partir_historique(dates, durees, len(dates))
            feats["machine"] = machine
            lignes.append(feats)
        etat_actuel = pd.DataFrame(lignes)
        X_actuel = self._encoder_features(etat_actuel, entrainement=False)

        resultat = etat_actuel[
            ["machine", "rang_panne", "intervalle_precedent_jours",
             "intervalle_moyen_jours", "pannes_recentes_30j"]
        ].copy()
        resultat = resultat.rename(columns={"rang_panne": "nb_pannes_historique"})

        derniere_panne = historique.groupby("machine")["date"].max()
        resultat = resultat.merge(derniere_panne.rename("derniere_panne"), on="machine", how="left")
        date_reference = historique["date"].max()
        resultat["jours_depuis_derniere_panne"] = (
            (date_reference - resultat["derniere_panne"]).dt.days
        )

        if self._classifieur is not None:
            proba = self._classifieur.predict_proba(X_actuel)[:, 1]
        elif self._regresseur is not None:
            # Pas de classifieur entraînable (une seule classe observée sur
            # l'historique disponible) -> risque déduit du délai prédit par
            # le régresseur plutôt que de renoncer à toute estimation.
            delai_brut = self._regresseur.predict(X_actuel)
            proba = np.clip(1 - (delai_brut / (2 * self.seuilJoursRisque)), 0.0, 1.0)
        else:
            proba = np.zeros(len(X_actuel))

        resultat["probabilite_panne_imminente"] = (proba * 100).round(1)
        resultat["niveau_risque"] = pd.cut(
            resultat["probabilite_panne_imminente"],
            bins=[-0.1, 25, 50, 75, 100],
            labels=NIVEAUX_RISQUE,
        )

        if self._regresseur is not None:
            # Délai calculé et affiché pour TOUTES les machines (pas
            # seulement Élevé/Critique) : c'est la durée prévue avant la
            # prochaine panne, résultat central attendu de ce prédicteur.
            delai_estime = np.round(np.clip(self._regresseur.predict(X_actuel), 0, None))
            resultat["delai_estime_jours"] = delai_estime
            resultat["date_prochaine_panne_estimee"] = resultat["derniere_panne"] + pd.to_timedelta(
                resultat["delai_estime_jours"], unit="D"
            )
        else:
            resultat["delai_estime_jours"] = np.nan
            resultat["date_prochaine_panne_estimee"] = pd.NaT

        resultat["nb_pannes_historique"] = resultat["nb_pannes_historique"].astype(int)
        resultat["jours_depuis_derniere_panne"] = resultat["jours_depuis_derniere_panne"].astype(int)

        colonnes_finales = [
            "machine", "probabilite_panne_imminente", "niveau_risque",
            "delai_estime_jours", "date_prochaine_panne_estimee",
            "nb_pannes_historique", "derniere_panne", "jours_depuis_derniere_panne",
        ]
        resultat = resultat[colonnes_finales].sort_values(
            "probabilite_panne_imminente", ascending=False
        ).reset_index(drop=True)
        logger.info("Prédiction de panne calculée pour %d machine(s).", len(resultat))
        return resultat

    # ------------------------------------------------------------------
    # Estimation par formule transparente (fichiers "instantané", sans
    # historique daté de pannes répétées : 1 ligne par machine, avec un
    # compteur d'usage et un taux de risque/défaut). PAS un modèle appris :
    # aucun délai réel n'est observable dans ce type de fichier (aucune
    # panne historique datée), donc aucun ML supervisé n'est possible.
    # ------------------------------------------------------------------
    def estimerParFormule(
        self,
        df: pd.DataFrame,
        colonne_machine: str,
        colonne_usage: str,
        colonne_taux: str,
        percentile_seuil_critique: float = PERCENTILE_SEUIL_CRITIQUE_DEFAUT,
    ) -> pd.DataFrame:
        """
        Estime, pour chaque machine, le temps restant (en heures, ou dans
        l'unité de `colonne_usage`) avant d'atteindre un niveau de risque
        jugé critique, par extrapolation LINÉAIRE du taux de dégradation
        observé :

          vitesse_degradation = taux_risque / compteur_usage
          seuil_critique       = percentile `percentile_seuil_critique` du
                                  taux de risque, calculé sur les machines
                                  du fichier (calibré sur les données
                                  réelles, pas une constante arbitraire)
          temps_restant_estime = (seuil_critique - taux_risque) / vitesse_degradation

        Retourne un DataFrame (1 ligne par machine), trié par temps restant
        estimé croissant (le plus urgent en premier) : machine,
        heures_fonctionnement, taux_risque_pct, temps_restant_estime_heures,
        niveau_risque. Vide si les colonnes fournies ne sont pas exploitables.
        """
        if (
            colonne_machine not in df.columns
            or colonne_usage not in df.columns
            or colonne_taux not in df.columns
        ):
            return pd.DataFrame()

        travail = pd.DataFrame({
            "machine": df[colonne_machine].astype(str).str.strip(),
            "usage": pd.to_numeric(df[colonne_usage], errors="coerce"),
            "taux": pd.to_numeric(df[colonne_taux], errors="coerce"),
        })
        travail = travail[
            travail["machine"].notna()
            & (travail["machine"] != "")
            & (travail["machine"].str.lower() != "nan")
        ]
        travail = travail.dropna(subset=["usage", "taux"])
        if travail.empty:
            return pd.DataFrame()

        # Une même machine peut apparaître plusieurs fois (ex. plusieurs
        # lignes/postes) : on retient la mesure la plus défavorable (taux de
        # risque le plus élevé), par prudence.
        travail = travail.sort_values("taux", ascending=False).drop_duplicates(
            "machine", keep="first"
        )

        seuil_critique = float(travail["taux"].quantile(percentile_seuil_critique))
        seuil_critique = max(seuil_critique, float(travail["taux"].max()))

        vitesse = travail["taux"] / travail["usage"].replace(0, np.nan)
        ecart = (seuil_critique - travail["taux"]).clip(lower=0)
        temps_restant = ecart / vitesse
        # Vitesse nulle (taux de risque à 0, aucune dégradation mesurable)
        # -> non estimable. Machine déjà au niveau critique -> 0 (immédiat).
        temps_restant = temps_restant.where(vitesse > 0)
        temps_restant = temps_restant.where(travail["taux"] < seuil_critique, 0.0)

        resultat = pd.DataFrame({
            "machine": travail["machine"],
            "heures_fonctionnement": travail["usage"],
            "taux_risque_pct": travail["taux"],
            "temps_restant_estime_heures": temps_restant.round(1),
        })
        resultat["niveau_risque"] = pd.cut(
            _normaliser_0_1(resultat["taux_risque_pct"]) * 100,
            bins=[-0.1, 25, 50, 75, 100],
            labels=NIVEAUX_RISQUE,
        )
        resultat = resultat.sort_values(
            "temps_restant_estime_heures", ascending=True, na_position="last"
        ).reset_index(drop=True)
        logger.info(
            "Estimation par formule (temps restant) calculée pour %d machine(s).", len(resultat)
        )
        return resultat

    # ------------------------------------------------------------------
    # Visualisations
    # ------------------------------------------------------------------
    def graphiqueClassementFormule(self, estimation: pd.DataFrame, top_n: int = 15):
        """Barres horizontales du temps restant estimé (le plus urgent en
        haut), colorées par niveau de risque. Les machines sans estimation
        (vitesse de dégradation nulle) sont ignorées dans ce graphique."""
        if estimation.empty:
            return None
        top = estimation.dropna(subset=["temps_restant_estime_heures"]).head(top_n).iloc[::-1]
        if top.empty:
            return None
        couleurs = [COULEURS_NIVEAU.get(str(n), TE_GRIS) for n in top["niveau_risque"]]

        fig, ax = plt.subplots(figsize=(9, max(4, 0.4 * len(top))))
        barres = ax.barh(
            top["machine"].astype(str), top["temps_restant_estime_heures"],
            color=couleurs, edgecolor=TE_ANTHRACITE,
        )
        ax.bar_label(barres, padding=3, fontsize=8, fmt="%.0f h")
        ax.set_xlabel("Temps restant estimé avant panne (heures)")
        ax.set_title(
            "Machines classées par urgence (estimation par formule)",
            color=TE_ANTHRACITE, fontweight="bold",
        )
        fig.tight_layout()
        return fig

    def genererResumeFormule(self, estimation: pd.DataFrame, top_n: int = 3) -> str:
        """Résumé automatique des machines les plus urgentes (estimation par
        formule)."""
        if estimation.empty:
            return (
                "Pas assez de données numériques exploitables (compteur "
                "d'usage + taux de risque) pour estimer un temps restant."
            )

        top = estimation.dropna(subset=["temps_restant_estime_heures"]).head(top_n)
        lignes = ["**Machines à surveiller en priorité (estimation par formule) :**", ""]
        for r in top.itertuples():
            lignes.append(
                f"- **{r.machine}** — {r.niveau_risque} : ~{r.temps_restant_estime_heures:.0f} h "
                f"restantes avant le seuil critique (taux de risque actuel "
                f"{r.taux_risque_pct:.1f} %, {r.heures_fonctionnement:.0f} h de fonctionnement)"
            )

        n_critique = int((estimation["niveau_risque"] == "Critique").sum())
        n_eleve = int((estimation["niveau_risque"] == "Élevé").sum())
        if n_critique:
            lignes.append("")
            lignes.append(
                f"{n_critique} machine(s) en niveau **Critique** : une intervention "
                "préventive rapprochée est recommandée."
            )
        elif n_eleve:
            lignes.append("")
            lignes.append(f"{n_eleve} machine(s) en niveau **Élevé** : à surveiller de près.")

        lignes.append("")
        lignes.append(
            "_Estimation par extrapolation linéaire du taux de dégradation "
            "observé (pas un modèle appris : ce fichier ne contient aucun "
            "historique daté de pannes permettant d'entraîner un modèle "
            "prédictif au sens strict)._"
        )
        return "\n".join(lignes)

    def graphiqueClassement(self, predictions: pd.DataFrame, top_n: int = 15):
        """Barres horizontales de la probabilité de panne imminente,
        machine la plus à risque en haut, colorées par niveau de risque."""
        if predictions.empty:
            return None
        top = predictions.head(top_n).iloc[::-1]
        couleurs = [COULEURS_NIVEAU.get(str(n), TE_GRIS) for n in top["niveau_risque"]]

        fig, ax = plt.subplots(figsize=(9, max(4, 0.4 * len(top))))
        barres = ax.barh(
            top["machine"].astype(str), top["probabilite_panne_imminente"],
            color=couleurs, edgecolor=TE_ANTHRACITE,
        )
        ax.bar_label(barres, padding=3, fontsize=8, fmt="%.0f%%")
        ax.set_xlabel("Probabilité de panne imminente (%)")
        ax.set_xlim(0, 105)
        ax.set_title(
            "Machines classées par risque de panne imminente (modèle prédictif)",
            color=TE_ANTHRACITE, fontweight="bold",
        )
        fig.tight_layout()
        return fig

    def graphiqueTendanceMachine(
        self,
        df: pd.DataFrame,
        colonne_machine: str,
        colonne_date: str,
        machine: str,
    ):
        """Courbe hebdomadaire du nombre de pannes pour UNE machine :
        permet d'inspecter le détail d'une machine à risque avant de
        planifier une intervention préventive."""
        sous = df[df[colonne_machine].astype(str).str.strip() == machine].copy()
        if sous.empty or colonne_date not in sous.columns:
            return None
        sous["date"] = pd.to_datetime(sous[colonne_date], errors="coerce")
        sous = sous.dropna(subset=["date"])
        if sous.empty:
            return None
        serie = sous.set_index("date").resample("W").size()
        if serie.empty:
            return None

        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(serie.index, serie.values, marker="o", color=TE_ORANGE, linewidth=2)
        ax.set_ylabel("Nombre de pannes / semaine")
        ax.set_title(f"Historique des pannes — {machine}", color=TE_ANTHRACITE, fontweight="bold")
        fig.autofmt_xdate()
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Résumé texte
    # ------------------------------------------------------------------
    def genererResume(self, predictions: pd.DataFrame, top_n: int = 3) -> str:
        """Résumé automatique des machines les plus à risque."""
        if predictions.empty:
            return (
                "Pas assez d'historique daté pour entraîner un modèle prédictif "
                "(il faut au moins quelques machines ayant enregistré 2 pannes "
                "ou plus)."
            )

        top = predictions.sort_values("delai_estime_jours", na_position="last").head(top_n)
        lignes = ["**Durée prévue avant la prochaine panne (modèle prédictif) :**", ""]
        for r in top.itertuples():
            if pd.notna(r.delai_estime_jours):
                details = f"~{int(r.delai_estime_jours)} jour(s) avant la prochaine panne prévue"
            else:
                details = "délai non estimable"
            details += f" (probabilité de panne imminente : {r.probabilite_panne_imminente:.0f} %)"
            if pd.notna(r.jours_depuis_derniere_panne):
                details += f", dernière panne il y a {int(r.jours_depuis_derniere_panne)} jour(s)"
            lignes.append(f"- **{r.machine}** — {r.niveau_risque} : {details}")

        n_critique = int((predictions["niveau_risque"] == "Critique").sum())
        n_eleve = int((predictions["niveau_risque"] == "Élevé").sum())
        if n_critique:
            lignes.append("")
            lignes.append(
                f"{n_critique} machine(s) en niveau **Critique** : une intervention "
                "préventive rapprochée est recommandée."
            )
        elif n_eleve:
            lignes.append("")
            lignes.append(f"{n_eleve} machine(s) en niveau **Élevé** : à surveiller de près.")

        return "\n".join(lignes)

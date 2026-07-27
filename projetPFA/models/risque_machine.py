"""
models/risque_machine.py
--------------------------
Score de risque par machine (maintenance prédictive) — module additionnel,
indépendant du pipeline ML de détection de lignes utiles/non utiles
(models/pipeline_principal.py, models/modele_ml.py).

Construit, à partir d'un journal d'interventions (1 ligne = 1 panne /
intervention), un classement des équipements par risque de panne, calculé
à partir de règles simples et transparentes (pas un modèle "boîte noire",
volontairement — fiable même avec peu de lignes par machine) :

  - fréquence des pannes (nombre d'interventions)
  - durée cumulée d'arrêt (si une colonne durée est disponible)
  - tendance récente : pannes des N derniers jours vs les N précédents
    (si une colonne date est disponible)
  - récence : temps écoulé depuis la dernière panne, approximant le MTBF
    (si une colonne date est disponible)

Chaque composante est normalisée entre 0 et 1 (min-max sur l'ensemble des
machines du fichier), puis combinée en un score pondéré sur 100. Si aucune
colonne date n'est fournie, les poids "tendance"/"récence" sont
automatiquement redistribués sur fréquence/durée pour que le score reste
interprétable sur 100.

Générique : ne dépend d'aucun nom de colonne fixe. Les colonnes
"machine" / "durée d'arrêt" / "date" sont fournies par l'appelant
(l'UI les propose via colonnes_machine_candidates / colonnes_date_candidates,
avec sélection manuelle possible en cas de mauvaise détection).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from models.logger import get_logger

logger = get_logger(__name__)

# Charte graphique TE Connectivity (cohérente avec les autres modules)
TE_ORANGE = "#FF8200"
TE_ORANGE_DARK = "#CC6500"
TE_ANTHRACITE = "#2B2B2B"
TE_GRIS = "#6A6A6A"

NIVEAUX_RISQUE = ["Faible", "Modéré", "Élevé", "Critique"]
COULEURS_NIVEAU = {
    "Faible": "#8FBF7F",
    "Modéré": "#FFD34D",
    "Élevé": "#FF9A33",
    "Critique": "#C0392B",
}


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
class AnalyseRisqueMachine:
    """Calcule et présente le score de risque par machine."""

    fenetreRecenteJours: int = 30
    poidsFrequence: float = 0.35
    poidsDureeCumulee: float = 0.25
    poidsTendance: float = 0.20
    poidsRecence: float = 0.20

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
            if serie.dtype == object:
                echantillon = serie.dropna().astype(str).head(30)
                if echantillon.empty:
                    continue
                parsed = pd.to_datetime(echantillon, errors="coerce")
                if parsed.notna().mean() >= 0.7:
                    candidates.append(col)
        candidates.sort(key=lambda c: 0 if "date" in c.lower() else 1)
        return candidates

    # ------------------------------------------------------------------
    # Calcul du score
    # ------------------------------------------------------------------
    def calculerScores(
        self,
        df: pd.DataFrame,
        colonne_machine: str,
        colonne_duree: Optional[str] = None,
        colonne_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Retourne un DataFrame (1 ligne par machine), trié par score de
        risque décroissant :
          machine, score_risque (0-100), niveau_risque, nb_pannes,
          duree_totale_min, duree_moyenne_min,
          [derniere_panne, jours_depuis_derniere_panne, mtbf_jours,
           pannes_recentes, pannes_precedentes, tendance_pct] (si date fournie).
        """
        if colonne_machine not in df.columns:
            return pd.DataFrame()

        travail = df[[colonne_machine]].copy()
        travail = travail.rename(columns={colonne_machine: "machine"})
        travail["machine"] = travail["machine"].astype(str).str.strip()
        travail = travail[
            travail["machine"].notna()
            & (travail["machine"] != "")
            & (travail["machine"].str.lower() != "nan")
        ]
        if travail.empty:
            return pd.DataFrame()

        if colonne_duree and colonne_duree in df.columns:
            travail["duree"] = pd.to_numeric(
                df.loc[travail.index, colonne_duree], errors="coerce"
            )
        else:
            travail["duree"] = np.nan

        date_dispo = bool(colonne_date and colonne_date in df.columns)
        if date_dispo:
            travail["date"] = pd.to_datetime(
                df.loc[travail.index, colonne_date], errors="coerce"
            )
            date_dispo = travail["date"].notna().any()

        # -- Fréquence et durée cumulée ---------------------------------
        agg = (
            travail.groupby("machine")
            .agg(
                nb_pannes=("machine", "count"),
                duree_totale_min=("duree", "sum"),
                duree_moyenne_min=("duree", "mean"),
            )
            .reset_index()
        )
        agg["duree_totale_min"] = agg["duree_totale_min"].fillna(0.0)
        agg["duree_moyenne_min"] = agg["duree_moyenne_min"].fillna(0.0)

        # -- Récence, tendance, MTBF (nécessitent une colonne date) -----
        if date_dispo:
            travail_dt = travail.dropna(subset=["date"])
            date_max = travail_dt["date"].max()
            seuil_recent = date_max - pd.Timedelta(days=self.fenetreRecenteJours)
            seuil_precedent = seuil_recent - pd.Timedelta(days=self.fenetreRecenteJours)

            derniere = travail_dt.groupby("machine")["date"].max()
            premiere = travail_dt.groupby("machine")["date"].min()
            agg = agg.merge(derniere.rename("derniere_panne"), on="machine", how="left")
            agg = agg.merge(premiere.rename("premiere_panne"), on="machine", how="left")
            agg["jours_depuis_derniere_panne"] = (date_max - agg["derniere_panne"]).dt.days

            etendue_jours = (agg["derniere_panne"] - agg["premiere_panne"]).dt.days
            diviseur = (agg["nb_pannes"] - 1).replace(0, np.nan)
            agg["mtbf_jours"] = np.where(agg["nb_pannes"] > 1, etendue_jours / diviseur, np.nan)

            pannes_recentes = (
                travail_dt[travail_dt["date"] > seuil_recent].groupby("machine").size()
            )
            pannes_precedentes = (
                travail_dt[
                    (travail_dt["date"] <= seuil_recent) & (travail_dt["date"] > seuil_precedent)
                ]
                .groupby("machine")
                .size()
            )
            agg = agg.merge(pannes_recentes.rename("pannes_recentes"), on="machine", how="left")
            agg = agg.merge(pannes_precedentes.rename("pannes_precedentes"), on="machine", how="left")
            agg["pannes_recentes"] = agg["pannes_recentes"].fillna(0).astype(int)
            agg["pannes_precedentes"] = agg["pannes_precedentes"].fillna(0).astype(int)
            agg["tendance_pct"] = np.where(
                agg["pannes_precedentes"] > 0,
                (agg["pannes_recentes"] - agg["pannes_precedentes"]) / agg["pannes_precedentes"] * 100,
                np.where(agg["pannes_recentes"] > 0, 100.0, 0.0),
            )
        else:
            agg["derniere_panne"] = pd.NaT
            agg["jours_depuis_derniere_panne"] = np.nan
            agg["mtbf_jours"] = np.nan
            agg["pannes_recentes"] = np.nan
            agg["pannes_precedentes"] = np.nan
            agg["tendance_pct"] = np.nan

        # -- Normalisation des composantes (0-1) -------------------------
        score_frequence = _normaliser_0_1(agg["nb_pannes"])
        score_duree = _normaliser_0_1(agg["duree_totale_min"])

        if date_dispo:
            score_tendance = _normaliser_0_1(agg["tendance_pct"].clip(lower=0))
            jours = agg["jours_depuis_derniere_panne"]
            jours_rempli = jours.fillna(jours.max() if jours.notna().any() else 0)
            score_recence = 1 - _normaliser_0_1(jours_rempli)  # récent = plus risqué
            poids_freq, poids_duree = self.poidsFrequence, self.poidsDureeCumulee
            poids_tendance, poids_recence = self.poidsTendance, self.poidsRecence
        else:
            # Pas de colonne date : on redistribue les poids tendance/récence
            # (non calculables) sur fréquence/durée, au prorata de leurs
            # poids d'origine, pour que le score reste comparable sur 100.
            score_tendance = pd.Series(0.0, index=agg.index)
            score_recence = pd.Series(0.0, index=agg.index)
            poids_base = self.poidsFrequence + self.poidsDureeCumulee
            poids_a_redistribuer = self.poidsTendance + self.poidsRecence
            if poids_base > 0:
                poids_freq = self.poidsFrequence + poids_a_redistribuer * (
                    self.poidsFrequence / poids_base
                )
                poids_duree = self.poidsDureeCumulee + poids_a_redistribuer * (
                    self.poidsDureeCumulee / poids_base
                )
            else:
                poids_freq, poids_duree = 0.5, 0.5
            poids_tendance, poids_recence = 0.0, 0.0

        agg["score_risque"] = (
            score_frequence * poids_freq
            + score_duree * poids_duree
            + score_tendance * poids_tendance
            + score_recence * poids_recence
        ) * 100
        agg["score_risque"] = agg["score_risque"].round(1)

        agg["niveau_risque"] = pd.cut(
            agg["score_risque"],
            bins=[-0.1, 25, 50, 75, 100],
            labels=NIVEAUX_RISQUE,
        )

        colonnes_ordre = [
            "machine", "score_risque", "niveau_risque",
            "nb_pannes", "duree_totale_min", "duree_moyenne_min",
        ]
        if date_dispo:
            colonnes_ordre += [
                "derniere_panne", "jours_depuis_derniere_panne", "mtbf_jours",
                "pannes_recentes", "pannes_precedentes", "tendance_pct",
            ]
        agg = agg[colonnes_ordre].sort_values("score_risque", ascending=False).reset_index(drop=True)
        logger.info("Score de risque calculé pour %d machine(s).", len(agg))
        return agg

    # ------------------------------------------------------------------
    # Visualisations
    # ------------------------------------------------------------------
    def graphiqueClassement(self, scores: pd.DataFrame, top_n: int = 15):
        """Barres horizontales du score de risque, machine la plus à
        risque en haut, colorées par niveau de risque."""
        if scores.empty:
            return None
        top = scores.head(top_n).iloc[::-1]
        couleurs = [COULEURS_NIVEAU.get(str(n), TE_GRIS) for n in top["niveau_risque"]]

        fig, ax = plt.subplots(figsize=(9, max(4, 0.4 * len(top))))
        barres = ax.barh(
            top["machine"].astype(str), top["score_risque"], color=couleurs, edgecolor=TE_ANTHRACITE
        )
        ax.bar_label(barres, padding=3, fontsize=8, fmt="%.0f")
        ax.set_xlabel("Score de risque (0-100)")
        ax.set_xlim(0, 105)
        ax.set_title("Classement des machines par risque", color=TE_ANTHRACITE, fontweight="bold")
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
    def genererResume(self, scores: pd.DataFrame, top_n: int = 3) -> str:
        """Résumé automatique des machines les plus à risque."""
        if scores.empty:
            return "Aucune donnée exploitable pour calculer un score de risque."

        top = scores.head(top_n)
        lignes = ["**Machines à surveiller en priorité :**", ""]
        for r in top.itertuples():
            details = f"{r.nb_pannes} panne(s) enregistrée(s)"
            if getattr(r, "duree_totale_min", 0):
                details += f", {r.duree_totale_min:.0f} min d'arrêt cumulé"
            jours = getattr(r, "jours_depuis_derniere_panne", np.nan)
            if pd.notna(jours):
                details += f", dernière panne il y a {int(jours)} jour(s)"
            lignes.append(
                f"- **{r.machine}** — score {r.score_risque:.0f}/100 ({r.niveau_risque}) : {details}"
            )

        n_critique = int((scores["niveau_risque"] == "Critique").sum())
        n_eleve = int((scores["niveau_risque"] == "Élevé").sum())
        if n_critique:
            lignes.append("")
            lignes.append(
                f"⚠️ {n_critique} machine(s) en niveau **Critique** : une intervention "
                "préventive rapprochée est recommandée."
            )
        elif n_eleve:
            lignes.append("")
            lignes.append(f"{n_eleve} machine(s) en niveau **Élevé** : à surveiller de près.")

        return "\n".join(lignes)

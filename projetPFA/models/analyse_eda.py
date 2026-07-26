"""
models/analyse_eda.py
-----------------------
Bloc 2 - ANALYSE EXPLORATOIRE (EDA).

Deux classes, conformes au diagramme de classes :
- StatistiqueDescriptive : statistiques d'une colonne numérique.
- AnalyseEDA : calcule les statistiques de tout le jeu de données,
  les valeurs manquantes, et prépare les figures (histogrammes, top
  catégories, Pareto) affichées ensuite par app.py.

Suppressions (demande utilisateur) :
- Plus de boxplots.
- Plus de graphique de répartition brute des variables catégorielles
  (afficherRepartitionCategorielle) : les catégories sont désormais
  uniquement analysées via afficherTopCategories / afficherPareto,
  croisées avec une colonne numérique.
- Plus de matrice/heatmap de corrélation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from config import LONGUEUR_MOYENNE_MAX_CATEGORIE
from models.jeu_donnees import JeuDonnees
from models.logger import get_logger

logger = get_logger(__name__)


@dataclass
class StatistiqueDescriptive:
    """Statistiques descriptives d'une colonne numérique."""

    colonne: str
    moyenne: float
    mediane: float
    somme: float
    ecartType: float
    pourcentageValeurManquante: float
    variance: float

    @classmethod
    def calculerMoyenne(cls, serie: pd.Series) -> float:
        return float(serie.mean())

    @classmethod
    def calculerMedianne(cls, serie: pd.Series) -> float:
        return float(serie.median())

    @classmethod
    def calculerEcartType(cls, serie: pd.Series) -> float:
        return float(serie.std())

    @classmethod
    def calculerVariance(cls, serie: pd.Series) -> float:
        return float(serie.var())

    @classmethod
    def depuis_serie(cls, nom_colonne: str, serie: pd.Series) -> "StatistiqueDescriptive":
        return cls(
            colonne=nom_colonne,
            moyenne=cls.calculerMoyenne(serie),
            mediane=cls.calculerMedianne(serie),
            somme=float(serie.sum()),
            ecartType=cls.calculerEcartType(serie),
            pourcentageValeurManquante=float(serie.isna().mean() * 100),
            variance=cls.calculerVariance(serie),
        )


class AnalyseEDA:
    """Analyse exploratoire complète d'un JeuDonnees."""

    def __init__(self) -> None:
        self.statistiques: list[StatistiqueDescriptive] = []
        self.valeursManquantes: dict[str, float] = {}

    # ------------------------------------------------------------------
    def calculerStatistiques(self, jeu: JeuDonnees) -> list[StatistiqueDescriptive]:
        """Calcule les statistiques descriptives pour chaque colonne numérique."""
        df = jeu.dataframe
        colonnes_numeriques = df.select_dtypes(include=[np.number]).columns

        self.statistiques = [
            StatistiqueDescriptive.depuis_serie(col, df[col]) for col in colonnes_numeriques
        ]
        self.valeursManquantes = {
            col: float(df[col].isna().mean() * 100) for col in df.columns
        }
        logger.info("EDA : %d colonnes numériques analysées", len(self.statistiques))
        return self.statistiques

    def tableauValeursManquantes(self, jeu: JeuDonnees) -> pd.DataFrame:
        """Retourne un tableau détaillé des valeurs manquantes par colonne :
        nombre de cellules vides ET pourcentage, trié par pourcentage
        décroissant (les colonnes les plus problématiques en premier)."""
        df = jeu.dataframe
        n = len(df)
        tableau = pd.DataFrame(
            {
                "colonne": df.columns,
                "nb_valeurs_manquantes": [int(df[c].isna().sum()) for c in df.columns],
            }
        )
        tableau["pourcentage_manquant"] = (
            (tableau["nb_valeurs_manquantes"] / n * 100).round(2) if n else 0.0
        )
        return tableau.sort_values("pourcentage_manquant", ascending=False).reset_index(drop=True)

    @staticmethod
    def _colonnes_numeriques_pertinentes(df: pd.DataFrame) -> list[str]:
        """Sélectionne les colonnes numériques réellement exploitables en
        histogramme/boxplot : on exclut les colonnes entièrement vides (que
        pandas type quand même en float64 quand aucune valeur n'est
        renseignée, ce qui les faisait apparaître à tort comme numériques).
        """
        numeriques = df.select_dtypes(include=[np.number]).columns
        return [c for c in numeriques if df[c].notna().any()]

    @classmethod
    def colonnes_numeriques_triees_par_pertinence(cls, df: pd.DataFrame) -> list[str]:
        """Trie les colonnes numériques par pertinence :
        - une vraie mesure continue est préférée à un identifiant numérique
          ou à une colonne à très haute cardinalité purement descriptive ;
        - on accorde aussi un bonus aux colonnes ayant un nombre raisonnable
          de valeurs distinctes, car elles sont plus susceptibles d'être
          réellement analytiques (durée, coût, quantité, etc.)."""
        cols_pertinentes = cls._colonnes_numeriques_pertinentes(df)

        def _score_col(col: str) -> float:
            serie = df[col].dropna()
            if serie.empty:
                return 0.0
            n = len(serie)
            nunique = serie.nunique(dropna=True)
            ratio_uniques = nunique / n if n else 0.0
            variance = float(serie.var(skipna=True) or 0.0)

            # Probable identifiant ou index numérique : très haute cardinalité.
            est_identifiant = ratio_uniques > 0.9 and nunique > 10
            penalite_identifiant = 0.2 if est_identifiant else 1.0

            # Bonus léger pour une cardinalité modérée (pas trop faible, pas
            # trop proche de l'unicité totale), et pénalité pour les colonnes
            # presque constantes.
            bonus_cardinalite = max(min(ratio_uniques, 0.5), 0.05)

            return variance * penalite_identifiant * bonus_cardinalite

        return sorted(cols_pertinentes, key=_score_col, reverse=True)

    @classmethod
    def colonne_temps_arret(cls, df: pd.DataFrame) -> Optional[str]:
        """Détecte une colonne numérique représentant un temps d'arrêt."""
        priorites = [
            "durée de l'arrêt",
            "duree de l'arret",
            "temps d'arrêt",
            "temps darrêt",
            "temps arret",
            "downtime",
            "temps",
            "arret",
            "arrêt",
            "stop",
        ]
        colonnes = cls._colonnes_numeriques_pertinentes(df)
        lower_cols = {col: col.lower() for col in colonnes}
        for pattern in priorites:
            for col, lower in lower_cols.items():
                if pattern in lower:
                    return col
        return colonnes[0] if colonnes else None

    def afficherHistogrammes(
        self,
        jeu: JeuDonnees,
        max_colonnes: int = 1,
        colonnes: Optional[list[str]] = None,
    ):
        """Retourne une figure matplotlib avec un histogramme par colonne
        numérique réellement continue (durée, quantités...), en ne gardant
        que la/les colonne(s) la/les plus "informative(s)" (cf.
        colonnes_numeriques_triees_par_pertinence). Les colonnes vides ou
        catégorielles mal typées sont ignorées (cf.
        _colonnes_numeriques_pertinentes).

        Si `colonnes` est fourni, il est utilisé tel quel plutôt que la
        sélection automatique par pertinence."""
        if colonnes is None:
            cols = self.colonnes_numeriques_triees_par_pertinence(jeu.dataframe)[:max_colonnes]
        else:
            cols = [c for c in colonnes if c in jeu.dataframe.columns]
        df = jeu.dataframe[cols]
        if df.empty:
            return None
        fig, axes = plt.subplots(
            nrows=(len(df.columns) + 2) // 3, ncols=min(3, len(df.columns)), figsize=(12, 8)
        )
        axes = np.array(axes).reshape(-1)
        for i, col in enumerate(df.columns):
            sns.histplot(df[col].dropna(), kde=True, ax=axes[i], color="#2c6e91")
            axes[i].set_title(col)
        for j in range(len(df.columns), len(axes)):
            axes[j].axis("off")
        fig.tight_layout()
        return fig

    @staticmethod
    def _colonne_semble_date(serie: pd.Series, nom: str) -> bool:
        """Détecte si une colonne textuelle semble représenter une date."""
        nom = nom.lower()
        if any(mot in nom for mot in ("date", "jour", "mois", "annee", "année", "timestamp")):
            return True
        serie = serie.dropna().astype(str)
        if serie.empty:
            return False
        parsed = pd.to_datetime(serie, errors="coerce")
        ratio_dates = parsed.notna().sum() / len(serie)
        return ratio_dates >= 0.75

    @staticmethod
    def colonnes_categorielles_pertinentes(
        df: pd.DataFrame,
        min_cardinalite: int = 2,
        longueur_moyenne_max: int = LONGUEUR_MOYENNE_MAX_CATEGORIE,
    ) -> list[str]:
        """Détecte GENERIQUEMENT les colonnes catégorielles "intéressantes"
        à visualiser : ni des colonnes constantes (une seule valeur), ni du
        texte libre (commentaires, descriptions).

        Pas de plafond de cardinalité, même relatif : un classement Pareto
        sert justement à classer des ENTITÉS NOMMÉES (machines, employés,
        produits...) qui peuvent très bien apparaître chacune une seule
        fois dans le fichier (ex. un export déjà agrégé "1 ligne = 1
        machine", cardinalité 100 %) — exclure ces colonnes au motif
        qu'"il y a trop de valeurs différentes" reviendrait à interdire
        précisément le cas d'usage recherché. Seul le texte libre (valeurs
        longues, du type commentaire/description) est écarté, via
        `longueur_moyenne_max`, car un encodage catégoriel classique le
        ferait exploser en centaines de colonnes sans valeur analytique.
        Ne dépend d'aucun nom de colonne particulier."""
        colonnes = []
        n = len(df)
        if n == 0:
            return colonnes
        for col in df.select_dtypes(include=["object", "category"]).columns:
            serie = df[col].dropna().astype(str)
            if serie.empty:
                continue
            n_uniques = df[col].nunique(dropna=True)
            if n_uniques < min_cardinalite:
                continue
            if serie.str.len().mean() > longueur_moyenne_max:
                continue
            if AnalyseEDA._colonne_semble_date(df[col], col):
                continue
            colonnes.append(col)
        return colonnes

    def calculerAgregations(
        self,
        jeu: JeuDonnees,
        colonne_numerique: Optional[str] = None,
        colonnes_groupe: Optional[list[str]] = None,
    ) -> dict[str, pd.DataFrame]:
        """Agrège une colonne numérique (au choix, ou détectée automatiquement)
        par catégorie, pour chaque colonne catégorielle "pertinente" détectée
        automatiquement (ou fournie). Généralise l'ancien calcul de KPI
        métier (temps d'arrêt par machine) à n'importe quel fichier."""
        df = jeu.dataframe
        resultats: dict[str, pd.DataFrame] = {}

        if colonne_numerique is None:
            cols_num = self._colonnes_numeriques_pertinentes(df)
            colonne_numerique = cols_num[0] if cols_num else None
        if colonne_numerique is None or colonne_numerique not in df.columns:
            return resultats

        if colonnes_groupe is None:
            colonnes_groupe = self.colonnes_categorielles_pertinentes(df)

        for col in colonnes_groupe:
            if col not in df.columns:
                continue
            agg = (
                df.groupby(col)[colonne_numerique]
                .agg(nb_lignes="count", somme="sum", moyenne="mean")
                .sort_values("somme", ascending=False)
            )
            if agg["somme"].sum():
                agg["pourcentage"] = agg["somme"] / agg["somme"].sum() * 100
            resultats[col] = agg
        return resultats

    def afficherTopCategories(
        self,
        jeu: JeuDonnees,
        colonne_numerique: str,
        colonne_groupe: str,
        top_n: int = 10,
    ):
        """Retourne une figure matplotlib : top N catégories (n'importe
        laquelle, ex. machine, client, produit...) par somme d'une colonne
        numérique quelconque."""
        df = jeu.dataframe
        if colonne_groupe not in df.columns or colonne_numerique not in df.columns:
            return None
        top = (
            df.groupby(colonne_groupe)[colonne_numerique]
            .sum()
            .sort_values(ascending=False)
            .head(top_n)
        )
        if top.empty:
            return None
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=top.values, y=top.index, ax=ax, color="#2c6e91")
        ax.set_xlabel(colonne_numerique)
        ax.set_title(f"Top {top_n} « {colonne_groupe} » par {colonne_numerique}")
        fig.tight_layout()
        return fig


    def afficherPareto(
        self,
        jeu: JeuDonnees,
        colonne_numerique: str,
        colonne_groupe: str,
        seuil_classe_a: float = 80.0,
        seuil_classe_b: float = 95.0,
    ):
        """Diagramme de Pareto GENERIQUE avec classement ABC : barres =
        contribution de chaque catégorie (somme de `colonne_numerique`,
        ordre décroissant), courbe = pourcentage cumulé. TOUTES les
        catégories sont affichées (aucune troncature de la "longue traîne")
        et regroupées en 3 classes selon leur poids dans le cumul :
          - Classe A : catégories cumulant jusqu'à `seuil_classe_a` % (80 % par défaut)
          - Classe B : de `seuil_classe_a` à `seuil_classe_b` % (95 % par défaut)
          - Classe C : le reste (longue traîne, au-delà de `seuil_classe_b` %)
        Les frontières de classes sont matérialisées par des lignes
        verticales pointillées et des étiquettes A/B/C, quel que soit le
        fichier importé (aucun nom de colonne supposé à l'avance)."""
        df = jeu.dataframe
        if colonne_groupe not in df.columns or colonne_numerique not in df.columns:
            return None

        agg = (
            df.groupby(colonne_groupe)[colonne_numerique]
            .sum()
            .sort_values(ascending=False)
        )
        agg = agg[agg > 0]
        if agg.empty:
            return None

        cumul_pct = agg.cumsum() / agg.sum() * 100
        n = len(agg)

        def _limite_classe(seuil: float) -> int:
            """Nombre de catégories nécessaires pour que le cumul atteigne
            `seuil` % (au moins 1, même si la 1ère catégorie dépasse déjà
            le seuil à elle seule)."""
            return min(int((cumul_pct < seuil).sum()) + 1, n)

        idx_a = _limite_classe(seuil_classe_a)
        idx_b = max(_limite_classe(seuil_classe_b), idx_a)

        largeur_fig = max(10, n * 0.35)
        fig, ax1 = plt.subplots(figsize=(largeur_fig, 5.5))
        barres = ax1.bar(
            range(n), agg.values, color="#FF8200", edgecolor="#CC6500"
        )
        if n <= 15:
            ax1.bar_label(barres, padding=3, fontsize=8)
        ax1.set_ylabel(colonne_numerique)
        ax1.set_xticks(range(n))
        ax1.set_xticklabels(
            agg.index.astype(str), rotation=60, ha="right",
            fontsize=7 if n > 25 else 9,
        )
        ax1.set_ylim(0, agg.values.max() * 1.35)

        ax2 = ax1.twinx()
        ax2.plot(range(n), cumul_pct.values, color="#c0392b", linewidth=2)
        ax2.set_ylabel("Pourcentage cumulé (%)")
        ax2.set_ylim(0, 115)

        # Frontières et étiquettes des classes A / B / C, à la manière d'un
        # classement ABC (bornes matérialisées par des lignes pointillées).
        frontieres = [0, idx_a, idx_b, n]
        libelles_classes = ["A", "B", "C"]
        couleurs_classes = ["#FFD34D", "#FFB347", "#FFF0DA"]
        y_zone = agg.values.max() * 1.2
        for depart, fin, libelle, couleur in zip(
            frontieres[:-1], frontieres[1:], libelles_classes, couleurs_classes
        ):
            if fin <= depart:
                continue
            centre = (depart + fin - 1) / 2
            ax1.text(
                centre, y_zone, libelle, ha="center", va="center", fontsize=12,
                fontweight="bold", color="#2B2B2B",
                bbox=dict(boxstyle="round,pad=0.3", facecolor=couleur, edgecolor="none"),
            )
            if fin < n:
                ax1.axvline(fin - 0.5, color="#c0392b", linestyle="--", linewidth=1.2)

        ax1.set_title(
            f"Diagramme de Pareto (classement ABC) — {colonne_groupe} par {colonne_numerique}"
        )
        fig.tight_layout()
        return fig


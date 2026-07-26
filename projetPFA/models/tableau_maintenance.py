"""
models/tableau_maintenance.py
------------------------------
Tableau de bord de suivi des maintenances préventives et prédictives.

Format du tableau de saisie :
    - 2 lignes fixes : "Interventions planifiées" et "Interventions réelles" ;
    - colonnes : les 12 mois de l'année ;
    - valeurs  : nombre d'interventions de maintenance.

À partir de ces deux lignes, le module calcule automatiquement, pour
chaque mois, le taux de réalisation (Réelles / Planifiées x 100), puis
les KPI annuels, les graphiques et le diagramme de Pareto des mois les
plus critiques (écarts planifié / réel).

Il est volontairement indépendant du pipeline de nettoyage de données
(models/pipeline_principal.py) : il consomme uniquement le DataFrame
saisi par l'utilisateur.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import matplotlib.pyplot as plt
import pandas as pd

from models.logger import get_logger

logger = get_logger(__name__)

# --------------------------------------------------------------------------
# Charte graphique TE Connectivity
# --------------------------------------------------------------------------
TE_ORANGE = "#FF8200"
TE_ORANGE_DARK = "#CC6500"
TE_ORANGE_LIGHT = "#FFD3A3"
TE_ANTHRACITE = "#2B2B2B"
TE_GRIS = "#6A6A6A"

MOIS_ANNEE = [
    "Janvier", "Février", "Mars", "Avril", "Mai", "Juin",
    "Juillet", "Août", "Septembre", "Octobre", "Novembre", "Décembre",
]

COLONNE_TYPE = "Intervention"
LIGNE_PLANIFIEES = "Total de maintenance planifiées"
LIGNE_REELLES = "Total de maintenance réelles"
LIGNE_TAUX = "PM %"
LIGNES_FIXES = [LIGNE_PLANIFIEES, LIGNE_REELLES]

SEUIL_OBJECTIF_DEFAUT = 95.0   # % à partir duquel un mois est "dans l'objectif"
SEUIL_ALERTE_DEFAUT = 90.0     # % en dessous duquel une recommandation est émise
SEUIL_PARETO_DEFAUT = 80.0     # % cumulé définissant les mois critiques (Pareto)


def dataframe_vide() -> pd.DataFrame:
    """Construit le tableau de saisie initial : 2 lignes fixes
    (planifiées / réelles), 12 colonnes mois, valeurs à 0."""
    data = {COLONNE_TYPE: LIGNES_FIXES}
    for mois in MOIS_ANNEE:
        data[mois] = [0, 0]
    return pd.DataFrame(data)


@dataclass
class TableauMaintenance:
    """Regroupe les calculs et visualisations du tableau de bord maintenance."""

    seuilObjectif: float = SEUIL_OBJECTIF_DEFAUT
    seuilAlerte: float = SEUIL_ALERTE_DEFAUT
    seuilPareto: float = SEUIL_PARETO_DEFAUT

    # ------------------------------------------------------------------
    # Préparation / normalisation des données saisies
    # ------------------------------------------------------------------
    def normaliser(self, df: pd.DataFrame) -> pd.DataFrame:
        """Garantit la présence des 2 lignes fixes et des 12 colonnes mois,
        avec des valeurs numériques."""
        df = df.copy()
        if COLONNE_TYPE not in df.columns:
            df[COLONNE_TYPE] = LIGNES_FIXES[: len(df)]
        df = df.set_index(COLONNE_TYPE)
        for mois in MOIS_ANNEE:
            if mois not in df.columns:
                df[mois] = 0
        df = df.reindex(LIGNES_FIXES, fill_value=0)
        df[MOIS_ANNEE] = df[MOIS_ANNEE].apply(pd.to_numeric, errors="coerce").fillna(0)
        return df.reset_index()

    def ligneTaux(self, df: pd.DataFrame) -> pd.DataFrame:
        """Construit la 3e ligne 'PM %' (Réel / Planifié), en lecture
        seule, à afficher directement sous le tableau de saisie —
        comme la ligne de formule dans le fichier Excel d'origine."""
        df_mois = self.calculerParMois(df)
        valeurs = {
            mois: (f"{taux:.0f}%" if planifie > 0 else "")
            for mois, planifie, taux in zip(
                df_mois["Mois"], df_mois["Planifiees"], df_mois["Taux"]
            )
        }
        ligne = {COLONNE_TYPE: LIGNE_TAUX, **valeurs}
        return pd.DataFrame([ligne])

    # ------------------------------------------------------------------
    # Étape 5 (suite) : calcul automatique des indicateurs
    # ------------------------------------------------------------------
    def calculerParMois(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transforme le tableau (2 lignes x 12 mois) en un tableau par
        mois (12 lignes), avec Planifiees, Realisees et Taux (%)."""
        df = self.normaliser(df)
        planifiees = df.loc[df[COLONNE_TYPE] == LIGNE_PLANIFIEES, MOIS_ANNEE].iloc[0]
        reelles = df.loc[df[COLONNE_TYPE] == LIGNE_REELLES, MOIS_ANNEE].iloc[0]

        df_mois = pd.DataFrame(
            {
                "Mois": MOIS_ANNEE,
                "Planifiees": planifiees.values,
                "Realisees": reelles.values,
            }
        )
        df_mois["Taux"] = df_mois.apply(
            lambda r: round((r["Realisees"] / r["Planifiees"]) * 100, 1)
            if r["Planifiees"] > 0
            else 0.0,
            axis=1,
        )
        return df_mois

    def calculerKPI(self, df_mois: pd.DataFrame) -> dict:
        """Calcule les KPI annuels à partir du tableau par mois (déjà muni de 'Taux')."""
        total_planifie = int(df_mois["Planifiees"].sum())
        total_realise = int(df_mois["Realisees"].sum())
        taux_global = round((total_realise / total_planifie) * 100, 1) if total_planifie else 0.0
        mois_objectif_atteint = int((df_mois["Taux"] >= self.seuilObjectif).sum())
        mois_sous_objectif = int((df_mois["Taux"] < self.seuilObjectif).sum())

        return {
            "total_planifie": total_planifie,
            "total_realise": total_realise,
            "taux_global": taux_global,
            "mois_objectif_atteint": mois_objectif_atteint,
            "mois_sous_objectif": mois_sous_objectif,
        }

    # ------------------------------------------------------------------
    # Étape 6 : visualisation des performances
    # ------------------------------------------------------------------
    def graphiquePlanifieRealise(self, df_mois: pd.DataFrame):
        """Histogramme comparant maintenances planifiées et réalisées, par mois."""
        fig, ax = plt.subplots(figsize=(9, 4.5))
        largeur = 0.38
        x = range(len(df_mois))
        ax.bar([i - largeur / 2 for i in x], df_mois["Planifiees"], width=largeur,
               label="Planifiées", color=TE_ORANGE_LIGHT, edgecolor=TE_ORANGE_DARK)
        ax.bar([i + largeur / 2 for i in x], df_mois["Realisees"], width=largeur,
               label="Réalisées", color=TE_ORANGE, edgecolor=TE_ORANGE_DARK)
        ax.set_xticks(list(x))
        ax.set_xticklabels(df_mois["Mois"], rotation=45, ha="right")
        ax.set_ylabel("Nombre d'interventions")
        ax.set_title("Interventions planifiées vs réalisées", color=TE_ANTHRACITE)
        ax.legend()
        fig.tight_layout()
        return fig

    def graphiqueEvolutionTaux(self, df_mois: pd.DataFrame):
        """Courbe de l'évolution mensuelle du taux de réalisation."""
        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.plot(df_mois["Mois"], df_mois["Taux"], marker="o", color=TE_ORANGE, linewidth=2)
        ax.axhline(self.seuilObjectif, color=TE_GRIS, linestyle="--",
                    label=f"Objectif ({self.seuilObjectif:.0f} %)")
        ax.set_ylabel("Taux de réalisation (%)")
        ax.set_title("Évolution mensuelle du taux de réalisation", color=TE_ANTHRACITE)
        ax.set_xticks(range(len(df_mois)))
        ax.set_xticklabels(df_mois["Mois"], rotation=45, ha="right")
        ax.legend()
        fig.tight_layout()
        return fig

    def graphiquePerformanceMensuelle(self, df_mois: pd.DataFrame):
        """Barres horizontales illustrant la performance mensuelle (taux)."""
        df_tri = df_mois.sort_values("Taux", ascending=True)
        couleurs = [TE_ORANGE if t >= self.seuilObjectif else TE_ANTHRACITE for t in df_tri["Taux"]]
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.barh(df_tri["Mois"], df_tri["Taux"], color=couleurs)
        ax.axvline(self.seuilObjectif, color=TE_GRIS, linestyle="--")
        ax.set_xlabel("Taux de réalisation (%)")
        ax.set_title("Performance mensuelle", color=TE_ANTHRACITE)
        fig.tight_layout()
        return fig

    def graphiqueRepartitionAnnuelle(self, kpi: dict):
        """Diagramme circulaire : interventions réalisées vs restantes sur l'année."""
        realise = kpi["total_realise"]
        restant = max(kpi["total_planifie"] - kpi["total_realise"], 0)
        if realise + restant == 0:
            return None
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.pie(
            [realise, restant],
            labels=["Réalisées", "Restantes"],
            autopct="%1.1f%%",
            colors=[TE_ORANGE, TE_ORANGE_LIGHT],
            startangle=90,
            wedgeprops={"edgecolor": "white"},
        )
        ax.set_title("Répartition annuelle", color=TE_ANTHRACITE)
        fig.tight_layout()
        return fig

    def graphiquePareto(self, df_mois: pd.DataFrame):
        """Diagramme de Pareto des écarts (Planifiées - Réalisées) par mois,
        permettant d'identifier les mois les plus critiques."""
        df = df_mois.copy()
        df["Ecart"] = (df["Planifiees"] - df["Realisees"]).clip(lower=0)
        df_tri = df.sort_values("Ecart", ascending=False)
        df_tri = df_tri[df_tri["Ecart"] > 0]
        if df_tri.empty:
            return None
        total_ecart = df_tri["Ecart"].sum()
        df_tri["CumulPct"] = df_tri["Ecart"].cumsum() / total_ecart * 100

        fig, ax1 = plt.subplots(figsize=(9.5, 5.5))

        barres = ax1.bar(
            df_tri["Mois"], df_tri["Ecart"], color=TE_ORANGE, edgecolor=TE_ORANGE_DARK
        )
        ax1.bar_label(barres, padding=3, fontsize=9, color=TE_ANTHRACITE)
        ax1.set_ylabel("Écart (interventions non réalisées)")
        ax1.set_xticks(range(len(df_tri)))
        ax1.set_xticklabels(df_tri["Mois"], rotation=45, ha="right")
        ax1.set_ylim(0, df_tri["Ecart"].max() * 1.25)

        ax2 = ax1.twinx()
        ax2.plot(
            df_tri["Mois"], df_tri["CumulPct"], color=TE_ANTHRACITE,
            marker="o", markerfacecolor=TE_ORANGE, markeredgecolor=TE_ANTHRACITE,
            linewidth=2,
        )
        for x, y in zip(range(len(df_tri)), df_tri["CumulPct"]):
            ax2.annotate(
                f"{y:.0f}%", (x, y), textcoords="offset points", xytext=(0, 8),
                ha="center", fontsize=9, color=TE_ANTHRACITE,
            )
        ax2.axhline(self.seuilPareto, color=TE_GRIS, linestyle="--", linewidth=1,
                    label=f"Seuil {self.seuilPareto:.0f} %")
        ax2.set_ylabel("Cumul (%)")
        ax2.set_ylim(0, 115)
        ax2.legend(loc="lower right")

        ax1.set_title("Diagramme de Pareto — mois critiques", color=TE_ANTHRACITE, fontweight="bold")
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Étape 7 : analyse automatique
    # ------------------------------------------------------------------
    def genererResume(self, df_mois: pd.DataFrame, kpi: dict) -> str:
        """Génère un résumé textuel automatique de la performance annuelle."""
        df_valide = df_mois[df_mois["Planifiees"] > 0]
        if df_valide.empty:
            return "Aucune donnée de maintenance saisie pour le moment."

        meilleurs = df_valide.sort_values("Taux", ascending=False).head(3)
        a_ameliorer = df_valide.sort_values("Taux", ascending=True).head(3)

        lignes = [
            f"- Total annuel planifié : **{kpi['total_planifie']}** interventions",
            f"- Total annuel réalisé : **{kpi['total_realise']}** interventions",
            f"- Taux global de réalisation : **{kpi['taux_global']} %**",
            f"- Mois ayant atteint l'objectif ({self.seuilObjectif:.0f} %) : "
            f"**{kpi['mois_objectif_atteint']}**",
            f"- Mois sous l'objectif : **{kpi['mois_sous_objectif']}**",
            "",
            "**Meilleurs mois :** "
            + ", ".join(f"{r.Mois} ({r.Taux} %)" for r in meilleurs.itertuples()),
            "**Mois à améliorer :** "
            + ", ".join(f"{r.Mois} ({r.Taux} %)" for r in a_ameliorer.itertuples()),
        ]

        if kpi["taux_global"] < self.seuilAlerte:
            lignes.append(
                ""
                f"Le taux global de réalisation ({kpi['taux_global']} %) est inférieur "
                f"à {self.seuilAlerte:.0f} % : il est recommandé de renforcer les "
                "maintenances préventives sur les mois identifiés ci-dessus."
            )

        return "\n".join(lignes)

    # ------------------------------------------------------------------
    # Étape 9 : export des résultats
    # ------------------------------------------------------------------
    def exporterExcel(self, df_saisi: pd.DataFrame, df_mois: pd.DataFrame, kpi: dict) -> bytes:
        """Génère un fichier Excel en mémoire (tableau saisi + détail par
        mois + KPI)."""
        buffer = io.BytesIO()
        df_kpi = pd.DataFrame(
            {
                "Indicateur": [
                    "Total annuel planifié",
                    "Total annuel réalisé",
                    "Taux global annuel (%)",
                    "Mois ayant atteint l'objectif",
                    "Mois sous l'objectif",
                ],
                "Valeur": [
                    kpi["total_planifie"],
                    kpi["total_realise"],
                    kpi["taux_global"],
                    kpi["mois_objectif_atteint"],
                    kpi["mois_sous_objectif"],
                ],
            }
        )
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            df_saisi.to_excel(writer, index=False, sheet_name="Saisie")
            df_mois.to_excel(writer, index=False, sheet_name="Detail_par_mois")
            df_kpi.to_excel(writer, index=False, sheet_name="KPI")
        logger.info("Export Excel du tableau de suivi des maintenances généré.")
        return buffer.getvalue()
def calculerParMoisDepuisFichier(self, df: pd.DataFrame) -> pd.DataFrame:
    """
    Construit automatiquement le tableau de maintenance à partir
    d'un fichier d'interventions.
    """

    if df.empty:
        return pd.DataFrame()

    # Vérifier que les colonnes existent
    if "Mois" not in df.columns:
        return pd.DataFrame()

    # Nombre d'interventions réalisées par mois
    realisees = (
        df.groupby("Mois")
        .size()
        .reindex(MOIS_ANNEE, fill_value=0)
    )

    # Si le fichier ne contient pas les interventions planifiées,
    # on considère que le planifié = réalisé.
    planifiees = realisees.copy()

    df_mois = pd.DataFrame({
        "Mois": MOIS_ANNEE,
        "Planifiees": planifiees.values,
        "Realisees": realisees.values,
    })

    df_mois["Taux"] = df_mois.apply(
        lambda r: round(
            (r["Realisees"] / r["Planifiees"]) * 100,
            1
        ) if r["Planifiees"] > 0 else 0,
        axis=1,
    )

    return df_mois
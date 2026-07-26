"""
models/fichier_excel.py
------------------------
Bloc 1 - IMPORTATION DES DONNEES.

Classe FichierExcel : représente le fichier Excel/CSV source et gère son
chargement, sa validation, et la création du JeuDonnees associé.

Version GENERIQUE : aucune feuille, ligne d'en-tête ou colonne n'est
supposée connue à l'avance. Le fichier peut être :
- un tableau "propre" (en-tête en première ligne) ;
- un export "brut" avec logo/titre/lignes vides avant le vrai tableau
  (l'en-tête est alors détectée automatiquement) ;
- un CSV ;
- un classeur Excel / export Google Sheets contenant PLUSIEURS feuilles :
  `listerFeuilles()` / `possedeMultiplesFeuilles()` permettent à l'UI de
  détecter ce cas et de proposer un sélecteur avant d'appeler
  `charger(nom_feuille=...)`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Optional

import pandas as pd

from config import MAX_LIGNES_SCAN_ENTETE
from models.jeu_donnees import JeuDonnees
from models.logger import get_logger

logger = get_logger(__name__)


class FichierExcelError(Exception):
    """Erreur levée quand le fichier ne peut pas être chargé ou validé."""


@dataclass
class FichierExcel:
    """Représente le fichier Excel/CSV importé par l'utilisateur."""

    idFichier: int
    nom: str
    url: str
    format: str = ".xlsx"
    dateImport: date = field(default_factory=date.today)
    nomFeuille: Optional[str] = None
    _dataframe: Optional[pd.DataFrame] = field(default=None, repr=False)
    _feuilles_disponibles: list[str] = field(default_factory=list, repr=False)

    # ------------------------------------------------------------------
    # Détection des feuilles (Excel / Google Sheets exporté en .xlsx)
    # ------------------------------------------------------------------
    def listerFeuilles(self) -> list[str]:
        """Retourne la liste des noms de feuilles du classeur Excel.

        Pour un CSV (une seule "feuille" implicite) ou en cas d'échec de
        lecture, retourne une liste vide : l'appelant (UI) doit alors
        considérer qu'il n'y a rien à choisir et charger normalement.
        """
        if self.url.lower().endswith(".csv"):
            self._feuilles_disponibles = []
            return self._feuilles_disponibles
        try:
            classeur = pd.ExcelFile(self.url)
            self._feuilles_disponibles = list(classeur.sheet_names)
        except Exception as exc:
            logger.warning("Impossible de lister les feuilles de %s : %s", self.nom, exc)
            self._feuilles_disponibles = []
        return self._feuilles_disponibles

    def possedeMultiplesFeuilles(self) -> bool:
        """True si le fichier contient plus d'une feuille : l'UI doit alors
        proposer un sélecteur à l'utilisateur avant l'import."""
        return len(self.listerFeuilles()) > 1

    # ------------------------------------------------------------------
    # 1. Import du fichier
    # ------------------------------------------------------------------
    def charger(self, nom_feuille: Optional[str] = None) -> pd.DataFrame:
        """Charge le fichier Excel/CSV en DataFrame pandas, quel que soit
        son contenu (colonnes inconnues à l'avance).

        `nom_feuille` : nom de la feuille à charger si le classeur en
        contient plusieurs (choisi par l'utilisateur dans l'UI, via
        `listerFeuilles()`). Si non fourni, on utilise `self.nomFeuille`
        si déjà défini, sinon la première feuille du classeur.
        """
        feuille_cible = nom_feuille or self.nomFeuille
        try:
            if self.url.lower().endswith(".csv"):
                df = pd.read_csv(self.url)
            else:
                df = self._lire_excel_avec_entete_auto(self.url, feuille_cible)
                self.nomFeuille = feuille_cible or (
                    self._feuilles_disponibles[0] if self._feuilles_disponibles else self.nomFeuille
                )

            # Supprime les lignes et colonnes totalement vides (fin de
            # tableau, cellules mergées, colonnes fantômes "Unnamed: X").
            df = df.dropna(how="all").dropna(axis=1, how="all")

            # Nettoie les noms de colonnes : espaces invisibles en fin de
            # libellé, doublons de noms ("Unnamed: 1" répété...).
            df.columns = [str(c).strip() for c in df.columns]
            df = self._dedupliquer_colonnes(df)

            self._dataframe = self._inferer_types(df)
            logger.info(
                "Fichier chargé : %s (%d lignes, %d colonnes)",
                self.nom, len(self._dataframe), len(self._dataframe.columns),
            )
            return self._dataframe
        except Exception as exc:  # pragma: no cover - remonté à l'UI
            logger.error("Erreur de chargement du fichier %s : %s", self.nom, exc)
            raise FichierExcelError(f"Impossible de charger le fichier : {exc}") from exc

    # Libellés d'en-tête de la colonne de regroupement d'un export de
    # TABLEAU CROISÉ DYNAMIQUE Excel ("PivotTable"), quelle que soit la
    # langue de l'installation Excel d'origine.
    _LIBELLES_ENTETE_TCD = ("row labels", "étiquettes de lignes")
    # Libellés de la ligne de total d'un tel export, à exclure des données
    # (ce n'est pas une catégorie, mais une somme de toutes les catégories).
    _LIBELLES_TOTAL_TCD = ("grand total", "total général", "total general")

    def _lire_excel_avec_entete_auto(
        self, chemin: str, nom_feuille: Optional[str] = None
    ) -> pd.DataFrame:
        """Détecte automatiquement la ligne d'en-tête réelle d'un fichier
        Excel quelconque, sans connaître à l'avance sa structure.

        Principe : on scanne les premières lignes (MAX_LIGNES_SCAN_ENTETE)
        et on retient celle qui ressemble le plus à une ligne d'en-tête,
        c'est-à-dire celle qui a le plus de cellules non vides, textuelles
        et de valeurs uniques (des libellés de colonnes, pas des données).

        Cas particulier détecté en priorité : un export de TABLEAU CROISÉ
        DYNAMIQUE Excel collé tel quel (titre, filtres "Mois (All)"/"Semaine
        (All)", puis "Row Labels"/"Étiquettes de lignes" comme véritable
        en-tête, suivi d'une ligne "Grand Total"). Ce format n'a qu'une
        seule ligne d'en-tête utile mais elle ne "ressemble" pas forcément
        à un en-tête au sens du score générique (peu de colonnes remplies) :
        on la repère donc explicitement par son libellé, et on exclut la
        ligne de total finale qui n'est pas une catégorie de données.

        `nom_feuille` : feuille à lire si le classeur en contient plusieurs.
        Si None, on lit la première feuille (comportement pandas par
        défaut, `sheet_name=0`), mais on renseigne au passage
        `self._feuilles_disponibles` pour que l'UI puisse toujours
        proposer un choix a posteriori.
        """
        feuilles = self.listerFeuilles()
        feuille = nom_feuille if nom_feuille is not None else (feuilles[0] if feuilles else 0)

        brut = pd.read_excel(
            chemin, sheet_name=feuille, header=None, nrows=MAX_LIGNES_SCAN_ENTETE + 5
        )
        limite = min(MAX_LIGNES_SCAN_ENTETE, len(brut))

        meilleure_ligne = 0
        meilleur_score = -1
        ligne_tcd = None
        for i in range(limite):
            ligne = brut.iloc[i]
            premiere_valeur = str(ligne.iloc[0]).strip().lower() if pd.notna(ligne.iloc[0]) else ""
            if premiere_valeur in self._LIBELLES_ENTETE_TCD:
                ligne_tcd = i
                break
            valeurs = [v for v in ligne if pd.notna(v)]
            non_vides = len(valeurs)
            if non_vides < 2:
                continue
            textuelles = sum(1 for v in valeurs if isinstance(v, str))
            uniques = len(set(str(v) for v in valeurs))
            score = non_vides + textuelles + uniques
            if score > meilleur_score:
                meilleur_score = score
                meilleure_ligne = i

        meilleure_ligne = ligne_tcd if ligne_tcd is not None else meilleure_ligne
        df = pd.read_excel(chemin, sheet_name=feuille, header=meilleure_ligne)
        df = self._completer_entetes_fusionnees(df, brut, meilleure_ligne)

        if ligne_tcd is not None:
            df = self._retirer_ligne_total_tcd(df)
        return df

    @classmethod
    def _retirer_ligne_total_tcd(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Retire la ligne "Grand Total"/"Total général" d'un export de
        tableau croisé dynamique : c'est une somme de toutes les catégories,
        pas une catégorie en soi, elle fausserait tout classement (Pareto,
        analyse croisée...) si elle restait mélangée aux données."""
        premiere_colonne = df.columns[0]
        est_total = (
            df[premiere_colonne].astype(str).str.strip().str.lower().isin(cls._LIBELLES_TOTAL_TCD)
        )
        # Une "catégorie" sans libellé (ligne creuse en fin d'export) n'est
        # pas une donnée exploitable.
        df = df[~est_total & df[premiere_colonne].notna()]
        return df.reset_index(drop=True)

    @staticmethod
    def _completer_entetes_fusionnees(
        df: pd.DataFrame, brut: pd.DataFrame, ligne_entete: int
    ) -> pd.DataFrame:
        """Complète les colonnes lues comme "Unnamed: X" par pandas lorsque
        leur libellé provient en réalité d'une cellule FUSIONNÉE
        verticalement avec une des lignes AU-DESSUS de l'en-tête retenu
        (cas fréquent sur des exports avec en-tête sur plusieurs niveaux :
        un identifiant/une machine dont le nom de colonne n'apparaît que
        sur la toute première ligne d'en-tête, les lignes intermédiaires
        ne portant que des sous-libellés comme "Plan"/"Actual" ou
        "FRQ"/"MIN" répétés par mois/semaine).

        pandas ne résout pas les fusions de cellules : pour chaque colonne
        non nommée, on remonte ligne par ligne (jusqu'au début du fichier)
        et on retient la première valeur non vide rencontrée. Sans ce
        complément, ces colonnes resteraient techniquement inexploitables
        ("Unnamed: X"), quelle que soit la profondeur de la fusion.
        """
        if ligne_entete == 0:
            return df
        nouvelles_colonnes = []
        for position, colonne in enumerate(df.columns):
            nouveau_nom = colonne
            if str(colonne).startswith("Unnamed") and position < brut.shape[1]:
                for ligne in range(ligne_entete - 1, -1, -1):
                    valeur_dessus = brut.iat[ligne, position]
                    if pd.notna(valeur_dessus):
                        nouveau_nom = str(valeur_dessus).strip()
                        break
            nouvelles_colonnes.append(nouveau_nom)
        df.columns = nouvelles_colonnes
        return df

    @staticmethod
    def _dedupliquer_colonnes(df: pd.DataFrame) -> pd.DataFrame:
        """Renomme les colonnes dont le libellé est dupliqué (ex. plusieurs
        colonnes "Unnamed: X" ou deux colonnes portant le même nom), pour
        éviter les conflits dans le reste du pipeline."""
        compteur: dict[str, int] = {}
        nouvelles: list[str] = []
        for c in df.columns:
            nom = str(c)
            if nom in compteur:
                compteur[nom] += 1
                nouvelles.append(f"{nom}_{compteur[nom]}")
            else:
                compteur[nom] = 0
                nouvelles.append(nom)
        df.columns = nouvelles
        return df

    @staticmethod
    def _inferer_types(df: pd.DataFrame) -> pd.DataFrame:
        """Convertit automatiquement les colonnes textuelles qui contiennent
        en réalité des nombres ou des dates, de façon générique (basée sur
        le contenu, jamais sur un nom de colonne particulier — sauf un
        indice ("date") pour limiter les faux positifs sur la conversion
        de dates, plus risquée que celle des nombres)."""
        for col in df.columns:
            if df[col].dtype != object:
                continue
            echantillon = df[col].dropna()
            if echantillon.empty:
                continue

            converti_num = pd.to_numeric(df[col], errors="coerce")
            if converti_num.notna().sum() >= 0.8 * len(echantillon):
                df[col] = converti_num
                continue

            if any(mot in str(col).lower() for mot in ("date", "jour", "mois", "année", "annee")):
                converti_date = pd.to_datetime(df[col], errors="coerce", dayfirst=True)
                if converti_date.notna().sum() >= 0.8 * len(echantillon):
                    df[col] = converti_date
        return df

    def verifierFormat(self) -> bool:
        """Vérifie que l'extension du fichier est supportée (.xlsx, .xls, .csv)."""
        extensions_valides = (".xlsx", ".xls", ".csv")
        ok = self.nom.lower().endswith(extensions_valides)
        if not ok:
            raise FichierExcelError(
                f"Format de fichier non supporté : {self.nom}. "
                f"Formats acceptés : {extensions_valides}"
            )
        return ok

    def verifierColonnes(self, colonnes_obligatoires: list[str] | None = None) -> bool:
        """Vérifie la présence de colonnes obligatoires, si l'utilisateur en
        a défini (vide par défaut : aucune colonne n'est imposée)."""
        if self._dataframe is None:
            raise FichierExcelError("Le fichier doit être chargé avant de vérifier les colonnes.")

        colonnes_attendues = colonnes_obligatoires or []
        if not colonnes_attendues:
            return True

        colonnes_presentes = set(self._dataframe.columns)
        manquantes = [c for c in colonnes_attendues if c not in colonnes_presentes]
        if manquantes:
            logger.warning("Colonnes obligatoires manquantes : %s", manquantes)
        return len(manquantes) == 0

    def creerJeuDonnees(self) -> JeuDonnees:
        """Construit l'objet JeuDonnees à partir du DataFrame chargé."""
        if self._dataframe is None:
            self.charger()
        jeu = JeuDonnees(
            idDataset=self.idFichier,
            dataframe=self._dataframe.copy(),
            dateImport=self.dateImport,
        )
        logger.info(
            "JeuDonnees créé : %d lignes, %d colonnes", jeu.getNbLignes(), jeu.getNbColonnes()
        )
        return jeu
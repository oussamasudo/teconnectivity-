"""
config.py
---------
Configuration centrale du projet.

Version GENERIQUE : le pipeline n'est plus câblé sur un fichier métier
particulier (ex. maintenance TE Connectivity). Il doit fonctionner sur
n'importe quel fichier Excel/CSV importé par l'utilisateur.
"""

from __future__ import annotations

# Chemins
OUTPUT_DIR = "outputs"
LOG_DIR = "logs"
DEFAULT_OUTPUT_NAME = "Donnees_Nettoyees.xlsx"
DEFAULT_AUDIT_NAME = "audit_report.xlsx"

# Colonnes obligatoires attendues dans le fichier source (Bloc 1 - Import).
# Vide par défaut : aucune colonne n'est imposée puisque l'outil doit
# s'adapter à n'importe quel fichier. L'utilisateur peut, s'il le souhaite,
# renseigner ici (ou depuis l'UI) une liste de colonnes qu'il juge
# obligatoires pour SON fichier ; le pipeline avertit sans bloquer si
# elles sont absentes.
COLONNES_OBLIGATOIRES: list[str] = []

# Nom de la colonne cible : si le fichier source la fournit déjà (valeurs
# "Utile"/"Non_utile" ou 0/1), elle est utilisée comme vérité terrain.
# Sinon elle est générée automatiquement à partir de règles génériques
# (voir models/modele_ml.py -> generer_labels_utilite).
COLONNE_CIBLE = "Utile"

# --------------------------------------------------------------------
# Règles génériques de détection des lignes "non utiles" (utilisées
# uniquement quand le fichier ne fournit pas déjà de colonne cible) :
#   - ligne trop incomplète (trop de cellules vides)
#   - ligne strictement dupliquée
#   - valeur numérique aberrante (méthode IQR) sur au moins une colonne
# --------------------------------------------------------------------
SEUIL_TAUX_MANQUANT = 0.5      # une ligne est "trop vide" si > 50% de cellules vides
FACTEUR_IQR_OUTLIER = 3.0      # multiplicateur de l'IQR pour détecter une valeur aberrante

# Détection des colonnes de "texte libre" (commentaires, descriptions...),
# exclues des features ML mais conservées dans les fichiers exportés.
SEUIL_CARDINALITE_TEXTE_LIBRE = 0.5   # % de valeurs quasi-uniques
LONGUEUR_MOYENNE_MAX_CATEGORIE = 30   # longueur moyenne (caractères) au-delà de laquelle une colonne texte est jugée "libre"

# Colonnes jugées non pertinentes à l'export final si quasi entièrement vides
SEUIL_COLONNE_VIDE_EXPORT = 0.9

# Nombre de lignes scannées pour détecter automatiquement la ligne d'en-tête
# réelle d'un fichier Excel mal formaté (logo, titres, lignes vides...).
MAX_LIGNES_SCAN_ENTETE = 15

# Pré-traitement
METHODE_NORMALISATION_DEFAUT = "standard"   # "standard" ou "minmax"
METHODE_ENCODAGE_DEFAUT = "onehot"          # "onehot" ou "label"

# Machine Learning (RandomForestClassifier par défaut)
RF_NB_ARBRES_DEFAUT = 200
RF_PROFONDEUR_MAX_DEFAUT = None
RF_RANDOM_STATE_DEFAUT = 42
TAILLE_TEST_DEFAUT = 0.25

# Export
FEUILLE_DONNEES_VALIDES = "Lignes_utiles"
FEUILLE_LIGNES_REJETEES = "Lignes_rejetees"

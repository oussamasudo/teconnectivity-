"""
app.py
------
Interface utilisateur (Streamlit) du pipeline complet, organisée en MENU
LATERAL : chaque étape du diagramme de séquence est une page indépendante,
accessible en un clic dans la barre latérale.

Charte graphique : orange (identité visuelle TE Connectivity), navigation
latérale réalisée avec "streamlit_option_menu" (icônes Bootstrap, pas
d'emoji), écran d'accueil illustré du logo.

1. Import du fichier
2. Analyse exploratoire (EDA)
3. Prétraitement
4. Machine Learning
5. Export du résultat
6. Restitution (Power BI, hors code)

Dépendance supplémentaire à installer :
    pip install streamlit-option-menu
"""

from __future__ import annotations

import base64
import re
import tempfile
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import requests
import streamlit as st
from streamlit_option_menu import option_menu

from config import (
    COLONNE_CIBLE,
    COLONNES_OBLIGATOIRES,
    DEFAULT_AUDIT_NAME,
    DEFAULT_OUTPUT_NAME,
    FEUILLE_DONNEES_VALIDES,
)
from models.analyse_eda import AnalyseEDA
from models.fichier_excel import FichierExcel, FichierExcelError
from models.pipeline_principal import PipelinePrincipal
from models.pretraitement import Pretraitement
from models.tableau_maintenance import COLONNE_TYPE, MOIS_ANNEE, TableauMaintenance, dataframe_vide
from utils.file_utils import sauvegarder_upload_temporaire
from utils.metrics import matrice_confusion_vers_dataframe

# --------------------------------------------------------------------------
# Palette TE Connectivity
# --------------------------------------------------------------------------
TE_ORANGE_50   = "#FFF8F0"
TE_ORANGE_100  = "#FFEBD6"
TE_ORANGE_200  = "#FFD3A3"
TE_ORANGE_300  = "#FFB86B"
TE_ORANGE_400  = "#FF9A33"
TE_ORANGE_500  = "#FF8200"
TE_ORANGE_600  = "#E67300"
TE_ORANGE_700  = "#CC6500"
TE_ORANGE_800  = "#A85200"
TE_ORANGE_900  = "#7A3B00"

# Compatibilité avec le reste du code
TE_ORANGE = TE_ORANGE_500
TE_ORANGE_DARK = TE_ORANGE_700
TE_ORANGE_LIGHT = TE_ORANGE_50

TE_ANTHRACITE = "#2B2B2B"
TE_GRIS = "#6A6A6A"

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
LOGO_FILE = BASE_DIR / "logo.png"

print("Logo path:", LOGO_FILE)
print("Logo exists:", LOGO_FILE.exists())
st.set_page_config(
    page_title="Nettoyage intelligent de données par Machine Learning",
    layout="wide",
)

# --------------------------------------------------------------------------
# CSS personnalisé — thème orange TE Connectivity
# --------------------------------------------------------------------------
st.markdown(
    f"""
    <style>
        /* Fond général */
        .stApp {{
            background-color: #FFFFFF;
            font-family: 'Segoe UI', sans-serif;
        }}

        /* En-tête (logo + nom de l'application), identique sur toutes les pages */
        .te-entete {{
            display: flex;
            align-items: center;
            gap: 14px;
            padding-bottom: 16px;
            margin-bottom: 20px;
            border-bottom: 1px solid {TE_ORANGE_200};
        }}
        .te-entete img {{
            height: 40px;
            width: auto;
        }}
        .te-entete-titre {{
            font-size: 20px;
            font-weight: 700;
            color: {TE_ANTHRACITE};
        }}

        /* Titres de page */
        h1, h2, h3 {{
            color: {TE_ANTHRACITE};
        }}

        /* Barre latérale */
        section[data-testid="stSidebar"] {{
            background: linear-gradient(180deg, {TE_ORANGE_LIGHT} 0%, #FFFFFF 55%);
            border-right: 1px solid #F0DCC4;
            min-width: 300px;
            width: 300px;
        }}
        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2 {{
            color: {TE_ORANGE_DARK};
        }}

        /* Boutons — base commune (taille, forme, transition) */
        div.stButton > button, div.stDownloadButton > button,
        div.stFormSubmitButton > button,
        section[data-testid="stFileUploaderDropzone"] button {{
            border-radius: 10px;
            padding: 10px 22px;
            font-weight: 600;
            font-size: 15px;
            transition: all 0.18s ease-in-out;
        }}

        /* Boutons principaux (primary + téléchargement + upload) */
        div.stButton > button[kind="primary"],
        div.stDownloadButton > button,
        div.stFormSubmitButton > button[kind="primary"],
        section[data-testid="stFileUploaderDropzone"] button {{
            background-color: {TE_ORANGE};
            color: #FFFFFF;
            border: none;
            box-shadow: 0 3px 10px rgba(255, 130, 0, 0.35);
        }}
        div.stButton > button[kind="primary"]:hover,
        div.stDownloadButton > button:hover,
        div.stFormSubmitButton > button[kind="primary"]:hover,
        section[data-testid="stFileUploaderDropzone"] button:hover {{
            background-color: {TE_ORANGE_DARK};
            color: #FFFFFF;
            box-shadow: 0 5px 14px rgba(255, 130, 0, 0.45);
            transform: translateY(-1px);
        }}
        div.stButton > button[kind="primary"]:active,
        div.stDownloadButton > button:active,
        section[data-testid="stFileUploaderDropzone"] button:active {{
            transform: translateY(0);
            box-shadow: 0 2px 6px rgba(255, 130, 0, 0.35);
        }}

        /* Boutons secondaires */
        div.stButton > button:not([kind="primary"]) {{
            background-color: #FFFFFF;
            border: 1.5px solid {TE_ORANGE};
            color: {TE_ORANGE_DARK};
        }}
        div.stButton > button:not([kind="primary"]):hover {{
            background-color: {TE_ORANGE_LIGHT};
            border-color: {TE_ORANGE_DARK};
            color: {TE_ORANGE_DARK};
            transform: translateY(-1px);
            box-shadow: 0 3px 8px rgba(43, 43, 43, 0.12);
        }}
        div.stButton > button:not([kind="primary"]):active {{
            transform: translateY(0);
            box-shadow: none;
        }}

        /* Bouton désactivé : pas d'effet de survol trompeur */
        div.stButton > button:disabled, div.stDownloadButton > button:disabled {{
            box-shadow: none;
            transform: none;
            opacity: 0.6;
        }}

        /* Barre de progression */
        div[data-testid="stProgress"] > div > div {{
            background-color: {TE_ORANGE};
        }}

        /* Métriques */
        div[data-testid="stMetric"] {{
            background-color: {TE_ORANGE_LIGHT};
            border: 1px solid {TE_ORANGE};
            border-radius: 8px;
            padding: 10px;
        }}
        div[data-testid="stMetricValue"] {{
            color: {TE_ORANGE_DARK};
        }}

        /* Onglets */
        button[data-baseweb="tab"] {{
            color: {TE_GRIS};
        }}
        button[data-baseweb="tab"][aria-selected="true"] {{
            color: {TE_ORANGE_DARK};
            border-bottom-color: {TE_ORANGE} !important;
        }}

        /* Messages de succès / info / alerte */
        div[data-testid="stAlert"] {{
            border-left: 4px solid {TE_ORANGE};
        }}

        /* --------------------------------------------------------------
           Neutralisation du bleu par défaut de Streamlit (radio, upload,
           champs de saisie, liens...) : tout accent doit être orange,
           notamment sur la page d'Import.
        -------------------------------------------------------------- */
        /* Bouton radio sélectionné ("Source du fichier") */
        div[data-baseweb="radio"] div[aria-checked="true"] > div:first-child {{
            border-color: {TE_ORANGE} !important;
            background-color: {TE_ORANGE} !important;
        }}
        div[data-baseweb="radio"] div:first-child {{
            border-color: {TE_GRIS};
        }}
        label[data-baseweb="radio"]:hover div:first-child {{
            border-color: {TE_ORANGE} !important;
        }}

        /* Case à cocher */
        div[data-baseweb="checkbox"] span[aria-checked="true"] {{
            background-color: {TE_ORANGE} !important;
            border-color: {TE_ORANGE} !important;
        }}

        /* Zone d'upload de fichier : bordure et icône */
        section[data-testid="stFileUploaderDropzone"] {{
            border-color: {TE_ORANGE_300} !important;
            background-color: {TE_ORANGE_50};
        }}
        section[data-testid="stFileUploaderDropzone"]:hover {{
            border-color: {TE_ORANGE} !important;
        }}
        section[data-testid="stFileUploaderDropzone"] svg {{
            fill: {TE_ORANGE_DARK} !important;
        }}
        div[data-testid="stFileUploaderFile"] svg {{
            fill: {TE_ORANGE_DARK} !important;
        }}

        /* Champs de saisie (texte, URL) et menus déroulants : focus orange */
        div[data-baseweb="input"]:focus-within,
        div[data-baseweb="base-input"]:focus-within,
        div[data-baseweb="select"] > div:focus-within,
        div[data-baseweb="textarea"]:focus-within {{
            border-color: {TE_ORANGE} !important;
            box-shadow: 0 0 0 1px {TE_ORANGE} !important;
        }}
        input:focus, textarea:focus {{
            border-color: {TE_ORANGE} !important;
            box-shadow: 0 0 0 1px {TE_ORANGE} !important;
        }}

        /* Options survolées/sélectionnées dans les listes déroulantes */
        li[aria-selected="true"], li:hover {{
            background-color: {TE_ORANGE_100} !important;
        }}

        /* Liens */
        a, a:visited {{
            color: {TE_ORANGE_DARK};
        }}
        a:hover {{
            color: {TE_ORANGE_700};
        }}

        /* Libellé de champ (ex. "Source du fichier") */
        .te-field-label {{
            font-size: 14px;
            font-weight: 600;
            color: {TE_ANTHRACITE};
            margin-bottom: 6px;
        }}

        /* Titre du menu de navigation */
        .te-nav-title {{
            font-size: 13px;
            font-weight: 700;
            letter-spacing: 2px;
            text-transform: uppercase;
            color: {TE_ORANGE_DARK};
            margin: 4px 0 12px 4px;
        }}
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# En-tête sobre : logo + nom de l'application alignés sur une seule ligne,
# identique sur toutes les pages (remplace l'ancien logo isolé + le grand
# bandeau centré à fond dégradé, incohérents avec le reste de l'interface
# alignée à gauche).
# --------------------------------------------------------------------------
_logo_base64 = base64.b64encode(Path(LOGO_FILE).read_bytes()).decode("utf-8")
st.markdown(
    f"""
    <div class="te-entete">
        <img src="data:image/png;base64,{_logo_base64}" alt="TE Connectivity" />
        <span class="te-entete-titre">Nettoyage intelligent des données par Machine Learning</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# Initialisation de l'état de session
# --------------------------------------------------------------------------
if "pipeline" not in st.session_state:
    st.session_state["pipeline"] = PipelinePrincipal()
if "etape_terminee" not in st.session_state:
    st.session_state["etape_terminee"] = 0  # 0 = rien fait, 5 = pipeline complet
if "maintenance_df" not in st.session_state:
    st.session_state["maintenance_df"] = dataframe_vide()
if "tableau_maintenance" not in st.session_state:
    st.session_state["tableau_maintenance"] = TableauMaintenance()

pipeline: PipelinePrincipal = st.session_state["pipeline"]
etape_terminee = st.session_state["etape_terminee"]
tableau_maintenance: TableauMaintenance = st.session_state["tableau_maintenance"]


def reinitialiser_pipeline_pour_nouvelle_source(pipeline: PipelinePrincipal) -> None:
    """Réinitialise tout l'état calculé en aval de l'import (EDA,
    prétraitement, modèle, rapport) chaque fois qu'une NOUVELLE source de
    données devient active (nouveau fichier/URL/Google Sheet importé, OU
    nouvelle saisie manuelle enregistrée). Empêche qu'une ancienne EDA, un
    ancien jeu prétraité, un ancien modèle entraîné ou d'anciens résultats
    (df_resultat, lignes ajoutées manuellement...) provenant d'une source
    précédente ne restent affichés ou utilisés ailleurs dans l'application
    (EDA, Prétraitement, Machine Learning, Export)."""
    pipeline.analyseEDA = AnalyseEDA()
    pipeline.pretraitement = Pretraitement(
        methodeNormalisation=pipeline.pretraitement.methodeNormalisation,
        methodeEncodage=pipeline.pretraitement.methodeEncodage,
    )
    pipeline._jeu_nettoye = None
    pipeline._jeu_pretraite = None
    pipeline._labels = None
    pipeline._colonnes_texte_libre = []
    pipeline.rapport_regles = {}
    pipeline.source_cible = ""
    pipeline.modele = None
    pipeline.rapport = None

    # Anciens résultats/caches dépendant de la source précédente : à purger
    # pour qu'aucune ancienne DataFrame ne puisse resurgir dans l'UI (ex. un
    # tableau de résultats ML, ou des lignes en cours de saisie complémentaire
    # calées sur les colonnes de l'ancien jeu de données).
    for cle in ("df_resultat",):
        st.session_state.pop(cle, None)

    # Toute étape au-delà de l'import redevient à refaire sur la nouvelle
    # source (Import terminé = 1, tout le reste doit être relancé) : c'est ce
    # qui empêche l'EDA (et le reste) d'afficher encore les résultats d'une
    # source précédente.
    st.session_state["etape_terminee"] = 1


ETAPES = [
    "Import du fichier",
    "Analyse exploratoire (EDA)",
    "Prétraitement",
    "Machine Learning",
    "Export du résultat",
    
]
ICONES_ETAPES = [
    "cloud-upload",
    "bar-chart",
    "sliders",
    "cpu",
    "cloud-download",
]

# --------------------------------------------------------------------------
# MENU LATERAL (streamlit_option_menu)
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<p class="te-nav-title">Navigation</p>', unsafe_allow_html=True)

    page_label = option_menu(
        menu_title=None,
        options=ETAPES,
        icons=ICONES_ETAPES,
        menu_icon="list",
        default_index=0,
        styles={
            "container": {
                "padding": "8px",
                "background-color": "#FFFFFF",
                "border-radius": "14px",
                "box-shadow": "0 2px 10px rgba(43, 43, 43, 0.08)",
            },
            "icon": {
                "color": TE_ORANGE_DARK,
                "font-size": "20px",
            },
            "nav-link": {
                "font-size": "16px",
                "font-weight": "500",
                "margin": "6px 0",
                "padding": "12px 14px",
                "border-radius": "10px",
                "color": TE_ANTHRACITE,
                "transition": "background-color 0.15s ease-in-out",
                "--hover-color": TE_ORANGE_100,
            },
            "nav-link-selected": {
                "background-color": TE_ORANGE,
                "color": "white",
                "font-weight": "600",
                "box-shadow": "0 2px 8px rgba(255, 130, 0, 0.35)",
            },
        },
    )
    page_num = ETAPES.index(page_label) + 1

    st.progress(etape_terminee / 5)

    if st.button("Recommencer le pipeline", use_container_width=True):
        st.session_state.clear()
        st.rerun()

# --------------------------------------------------------------------------
# Titre de page — identique sur toutes les pages (y compris la toute
# première, avant import), pour une présentation homogène.
# --------------------------------------------------------------------------
st.title(ETAPES[page_num - 1])


EXTENSIONS_AUTORISEES = ("xlsx", "xls", "csv")


class TelechargementURLError(Exception):
    """Erreur levée lorsque le fichier pointé par l'URL est invalide ou inaccessible."""


def _extension_depuis_url(url: str) -> str:
    """Extrait l'extension (sans point) du chemin de l'URL, en minuscules."""
    chemin = urlparse(url).path
    suffixe = Path(chemin).suffix.lower().lstrip(".")
    return suffixe


def telecharger_fichier_url(url: str) -> tuple[str, str]:
    """
    Télécharge un fichier CSV/Excel depuis une URL, vérifie que son
    extension est autorisée ET que son contenu est réellement lisible
    dans le format annoncé, puis le sauvegarde dans un fichier temporaire.

    Retourne (chemin_local, nom_fichier).
    Lève TelechargementURLError si le format n'est pas respecté.
    """
    url = url.strip()
    if not url:
        raise TelechargementURLError("Veuillez saisir une URL.")

    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise TelechargementURLError("L'URL doit commencer par http:// ou https://.")

    extension = _extension_depuis_url(url)
    if extension not in EXTENSIONS_AUTORISEES:
        raise TelechargementURLError(
            f"Extension '{extension or 'inconnue'}' non supportée. "
            f"Extensions autorisées : {', '.join(EXTENSIONS_AUTORISEES)}."
        )

    try:
        reponse = requests.get(url, timeout=30, stream=True)
        reponse.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise TelechargementURLError(f"Échec du téléchargement : {e}") from e

    # Sauvegarde locale dans un fichier temporaire portant la bonne extension
    nom_fichier = Path(parsed.path).name or f"fichier_url.{extension}"
    suffixe = f".{extension}"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffixe) as tmp:
        for bloc in reponse.iter_content(chunk_size=8192):
            tmp.write(bloc)
        chemin_local = tmp.name

    # Vérification stricte : le contenu doit vraiment être lisible dans
    # le format annoncé par l'extension (évite les pages HTML d'erreur,
    # fichiers corrompus, mauvais Content-Type, etc.)
    try:
        if extension == "csv":
            pd.read_csv(chemin_local, nrows=5)
        else:  # xlsx / xls
            pd.read_excel(chemin_local, nrows=5)
    except Exception as e:
        Path(chemin_local).unlink(missing_ok=True)
        raise TelechargementURLError(
            f"Le fichier téléchargé ne respecte pas le format '.{extension}' attendu "
            f"(fichier corrompu, page d'erreur, ou mauvais type de contenu). Détail : {e}"
        ) from e

    return chemin_local, nom_fichier


def _id_google_sheet_depuis_url(url: str) -> str:
    """Extrait l'ID du document à partir d'un lien de partage Google Sheets
    (ex. https://docs.google.com/spreadsheets/d/<ID>/edit#gid=0)."""
    m = re.search(r"/spreadsheets/d/([a-zA-Z0-9-_]+)", url)
    if not m:
        raise TelechargementURLError(
            "Lien Google Sheets invalide : impossible d'y trouver l'identifiant "
            "du document (format attendu : "
            "https://docs.google.com/spreadsheets/d/ID/edit)."
        )
    return m.group(1)


def _gid_depuis_url(url: str) -> str:
    """Extrait l'identifiant de feuille (gid) s'il est présent dans l'URL,
    sinon retourne '0' (première feuille par défaut)."""
    m = re.search(r"gid=([0-9]+)", url)
    return m.group(1) if m else "0"


def telecharger_google_sheet(url: str) -> tuple[str, str]:
    """
    Télécharge les données d'un Google Sheet à partir de son lien de
    partage, en le convertissant en lien d'export CSV public, sans que
    l'utilisateur ait besoin de télécharger lui-même le fichier.

    Retourne (chemin_local, nom_fichier). Lève TelechargementURLError si le
    document n'est pas accessible (permissions insuffisantes, lien invalide,
    document supprimé...).
    """
    url = (url or "").strip()
    if not url:
        raise TelechargementURLError("Veuillez saisir un lien Google Sheets.")

    id_document = _id_google_sheet_depuis_url(url)
    gid = _gid_depuis_url(url)
    lien_export = (
        f"https://docs.google.com/spreadsheets/d/{id_document}/export"
        f"?format=csv&gid={gid}"
    )

    try:
        reponse = requests.get(lien_export, timeout=30)
        reponse.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise TelechargementURLError(f"Échec du téléchargement : {e}") from e

    # Google renvoie une page HTML de connexion (pas un CSV) quand le
    # document n'est pas partagé publiquement : on le détecte pour donner
    # un message d'erreur clair plutôt qu'un DataFrame illisible.
    contenu = reponse.content
    if b"<html" in contenu[:1000].lower():
        raise TelechargementURLError(
            "Ce Google Sheet n'est pas accessible : vérifie que le partage "
            "est activé (« Partager → Toute personne disposant du lien "
            "peut lire »)."
        )

    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        tmp.write(contenu)
        chemin_local = tmp.name

    try:
        pd.read_csv(chemin_local, nrows=5)
    except Exception as e:
        Path(chemin_local).unlink(missing_ok=True)
        raise TelechargementURLError(
            f"Le contenu récupéré n'est pas un tableau CSV valide. Détail : {e}"
        ) from e

    return chemin_local, f"google_sheet_{id_document}.csv"


# --------------------------------------------------------------------------
# 1. IMPORT DU FICHIER
# --------------------------------------------------------------------------
if page_num == 1:
    if "mode_import" not in st.session_state:
        st.session_state["mode_import"] = "Fichier local"

    st.markdown('<p class="te-field-label">Source du fichier</p>', unsafe_allow_html=True)
    options_import = ["Fichier local", "URL", "Saisie manuelle"]
    cols_mode_import = st.columns(len(options_import))
    for col, option in zip(cols_mode_import, options_import):
        with col:
            est_actif = st.session_state["mode_import"] == option
            if st.button(
                option,
                type="primary" if est_actif else "secondary",
                use_container_width=True,
                key=f"btn_mode_import_{option}",
            ):
                st.session_state["mode_import"] = option
                st.rerun()
    mode_import = st.session_state["mode_import"]

    upload = None
    url_fichier = None

    if mode_import == "Fichier local":
        upload = st.file_uploader(
            "Fichier Excel/CSV (n'importe quel fichier de données tabulaires)",
            type=list(EXTENSIONS_AUTORISEES),
        )
    elif mode_import == "URL":
        url_fichier = st.text_input(
            "URL du fichier (lien direct .xlsx/.csv, ou lien de partage Google Sheets)",
         
        )
        st.caption(
            "Pour un Google Sheet, colle simplement son lien de partage : il "
            "doit être accessible via **Partager → Toute personne disposant "
            "du lien peut lire**."
        )

    if mode_import in ("Fichier local", "URL"):
        colonnes_obligatoires_utilisateur = COLONNES_OBLIGATOIRES

        bouton_libelle = "Importer le fichier" if mode_import == "Fichier local" else "Importer depuis l'URL"
        declencheur = (upload is not None) if mode_import == "Fichier local" else bool(url_fichier and url_fichier.strip())

        def _finaliser_import(fichier: FichierExcel) -> None:
            """Termine l'import (charge le fichier via le pipeline, purge
            l'état d'une éventuelle source précédente, affiche les messages
            d'info/avertissement). Factorisé car appelé aussi bien à
            l'import direct (une seule feuille) qu'après confirmation du
            choix de feuille (fichier multi-feuilles)."""
            pipeline.importerDonnees(fichier)
            reinitialiser_pipeline_pour_nouvelle_source(pipeline)
            st.session_state["source_donnees_active"] = "fichier"
            st.session_state["maintenance_df"] = dataframe_vide()
            st.session_state.pop("import_feuilles_en_attente", None)
            feuille_info = f" (feuille « {fichier.nomFeuille} »)" if fichier.nomFeuille else ""
            st.info(
                f"Fichier importé{feuille_info} : {pipeline.jeuDonnees.getNbLignes()} lignes, "
                f"{pipeline.jeuDonnees.getNbColonnes()} colonnes."
            )
            colonnes_ok = fichier.verifierColonnes(colonnes_obligatoires_utilisateur)
            if not colonnes_ok:
                colonnes_manquantes = [
                    c for c in colonnes_obligatoires_utilisateur
                    if c not in pipeline.jeuDonnees.getColonnes()
                ]
                st.warning(
                    f"Colonnes obligatoires absentes : {colonnes_manquantes}. "
                    "Le pipeline continue malgré tout, en restant tolérant."
                )
            if pipeline.jeuDonnees.getNbLignes() == 0:
                st.warning(
                    "Le fichier importé ne contient aucune ligne de données "
                    "(feuille vide ou modèle non rempli). Ajoute des lignes "
                    "avant de lancer le pipeline."
                )

        if declencheur and st.button(bouton_libelle, type="primary"):
            try:
                if mode_import == "Fichier local":
                    chemin = sauvegarder_upload_temporaire(upload)
                    nom = upload.name
                elif "docs.google.com/spreadsheets" in url_fichier:
                    # Détection automatique d'un lien de partage Google
                    # Sheets au sein de la section URL générique.
                    chemin, nom = telecharger_google_sheet(url_fichier)
                else:
                    chemin, nom = telecharger_fichier_url(url_fichier)

                fichier_sonde = FichierExcel(idFichier=1, nom=nom, url=chemin)
                feuilles = fichier_sonde.listerFeuilles()

                if len(feuilles) > 1:
                    # Plusieurs feuilles détectées (classeur Excel ou export
                    # multi-feuilles) : on ne finalise pas l'import tout de
                    # suite, l'utilisateur doit d'abord choisir la feuille à
                    # traiter via le sélecteur affiché juste en dessous.
                    st.session_state["import_feuilles_en_attente"] = {
                        "chemin": chemin, "nom": nom, "feuilles": feuilles,
                    }
                else:
                    _finaliser_import(fichier_sonde)
            except (FichierExcelError, TelechargementURLError) as e:
                st.error(str(e))

        # ------------------------------------------------------------
        # Sélecteur de feuille, affiché uniquement si le fichier importé
        # (ou l'URL/Google Sheet téléchargé) contient plusieurs feuilles.
        # L'import se déclenche automatiquement dès qu'une feuille est
        # choisie dans la liste : pas de bouton de confirmation séparé,
        # l'aperçu des données apparaît dès ce premier choix.
        # ------------------------------------------------------------
        en_attente = st.session_state.get("import_feuilles_en_attente")
        if en_attente:
            st.info(
                f"Le fichier « {en_attente['nom']} » contient plusieurs feuilles : "
                "choisis celle à importer, l'import se lance automatiquement."
            )
            feuille_choisie = st.selectbox(
                "Feuille à importer", en_attente["feuilles"], key="import_feuille_select"
            )
            cle_feuille_traitee = (en_attente["chemin"], feuille_choisie)
            if st.session_state.get("import_feuille_traitee") != cle_feuille_traitee:
                try:
                    fichier = FichierExcel(
                        idFichier=1,
                        nom=en_attente["nom"],
                        url=en_attente["chemin"],
                        nomFeuille=feuille_choisie,
                    )
                    _finaliser_import(fichier)
                    # _finaliser_import() purge "import_feuilles_en_attente" :
                    # on le restaure pour garder le sélecteur disponible si
                    # l'utilisateur veut changer de feuille par la suite.
                    st.session_state["import_feuilles_en_attente"] = en_attente
                    st.session_state["import_feuille_traitee"] = cle_feuille_traitee
                except FichierExcelError as e:
                    st.error(str(e))

        if not declencheur and etape_terminee == 0:
            st.warning("Veuillez charger un dataset (fichier ou URL) pour démarrer le pipeline.")

    else:
        # ------------------------------------------------------------
        # Saisie manuelle : tableau de suivi des maintenances avec 2
        # LIGNES fixes — interventions planifiées et interventions
        # réelles — et une COLONNE par mois. Cette page ne contient QUE
        # la saisie : l'analyse (KPI, graphiques, Pareto) est affichée
        # exclusivement dans la page « Analyse exploratoire (EDA) »,
        # onglet « Suivi des maintenances ».
        # ------------------------------------------------------------
        
        column_config_maintenance = {
            COLONNE_TYPE: st.column_config.TextColumn(COLONNE_TYPE, disabled=True),
        }
        for mois in MOIS_ANNEE:
            column_config_maintenance[mois] = st.column_config.NumberColumn(
                mois, min_value=0, step=1
            )

        # IMPORTANT (correction du bug « la première saisie n'est jamais
        # enregistrée ») : `st.data_editor` retourne, à CHAQUE exécution du
        # script, l'état COURANT du tableau — y compris la modification que
        # l'utilisateur vient de faire, même lors du run déclenché par le
        # clic sur "Enregistrer". Le bug initial venait du fait que le bouton
        # relisait l'ANCIENNE valeur (`st.session_state["maintenance_df"]`,
        # non encore mise à jour) au lieu de la valeur FRAICHEMENT retournée
        # par le widget (`df_maintenance_edite`). En sauvegardant toujours
        # cette dernière — jamais une copie d'un run précédent — un seul
        # clic suffit désormais, dès la première saisie.
        df_maintenance_edite = st.data_editor(
            st.session_state["maintenance_df"],
            column_config=column_config_maintenance,
            num_rows="fixed",
            hide_index=True,
            use_container_width=True,
            key="maintenance_editor",
        )

        # Ligne "PM %" calculée automatiquement (comme la formule Excel
        # Réel / Planifié), recalculée à chaque frappe pour un retour
        # visuel immédiat, AVANT même de cliquer sur "Enregistrer".
        st.dataframe(
            tableau_maintenance.ligneTaux(df_maintenance_edite),
            hide_index=True,
            use_container_width=True,
        )

        col_enregistrer, col_statut = st.columns([1, 3])
        with col_enregistrer:
            enregistrer_clique = st.button(
                "Enregistrer", type="primary", key="btn_enregistrer_maintenance",
                use_container_width=True,
            )
        if enregistrer_clique:
            # On sauvegarde explicitement `df_maintenance_edite` (la valeur
            # retournée par le widget PENDANT CE MEME RUN), pas une valeur
            # mise en cache d'un run antérieur : c'est ce qui garantit que
            # la donnée saisie est prise en compte dès le premier clic.
            st.session_state["maintenance_df"] = df_maintenance_edite.copy()

            df_mnt_mois_saisie = tableau_maintenance.calculerParMois(df_maintenance_edite)
            if df_mnt_mois_saisie["Planifiees"].sum() > 0:
                # La saisie manuelle devient la source de données PRINCIPALE
                # du pipeline : elle remplace intégralement tout jeu de
                # données précédemment chargé (fichier local, URL, Google
                # Sheet), et tout résultat déjà calculé en aval (EDA,
                # prétraitement, modèle, export) est purgé pour ne conserver
                # aucune trace de l'ancienne source.
                pipeline.importerDonneesManuelles(df_mnt_mois_saisie, nom="Saisie manuelle")
                reinitialiser_pipeline_pour_nouvelle_source(pipeline)
                st.session_state["source_donnees_active"] = "saisie_manuelle"
                with col_statut:
                    st.warning(
                        "Saisie manuelle enregistrée comme source de données "
                        f"active du pipeline ({pipeline.jeuDonnees.getNbLignes()} "
                        f"mois, {pipeline.jeuDonnees.getNbColonnes()} colonnes). "
                        "Consulte l'onglet « Suivi des maintenances » dans la "
                        "page Analyse exploratoire (EDA)."
                    )
            else:
                with col_statut:
                    st.warning(
                        "Renseigne au moins un mois d'interventions planifiées "
                        "pour que la saisie devienne la source de données "
                        "active du pipeline."
                    )

        if st.session_state.get("source_donnees_active") == "saisie_manuelle":
            st.warning(
                "La saisie manuelle est actuellement la source de données "
                "active du pipeline."
            )

    if mode_import != "Saisie manuelle" and etape_terminee >= 1 and pipeline.jeuDonnees is not None:
        st.divider()
        st.subheader("Aperçu des données importées")
        st.dataframe(pipeline.jeuDonnees.dataframe.head(50), use_container_width=True)

 
        st.caption("Passe à l'étape Analyse exploratoire (EDA) dans le menu à gauche.")


# --------------------------------------------------------------------------
# 2. ANALYSE EXPLORATOIRE (EDA)
# --------------------------------------------------------------------------
elif page_num == 2:
    if etape_terminee < 1:
        st.warning("Réalise d'abord l'étape Import du fichier.")
    else:
        if st.button("Lancer l'analyse exploratoire", type="primary"):
            pipeline.analyserDonnees()
            st.session_state["etape_terminee"] = max(st.session_state["etape_terminee"], 2)

        if st.session_state["etape_terminee"] >= 2:
            eda = pipeline.analyseEDA

            # ------------------------------------------------------------
            # Navigation simplifiée à 3 onglets seulement :
            #   - Vue d'ensemble    : TOUTES les analyses générales (stats,
            #     valeurs manquantes, distributions,
            #     diagramme de Pareto global, indicateurs) ;
            #   - Analyse croisée   : uniquement les croisements entre deux
            #     variables (numérique x catégoriel) et leurs graphiques ;
            #   - Suivi des maintenances : indicateurs de maintenance
            #     exclusivement (TP, TR, PM %, taux, objectifs, export).
            # ------------------------------------------------------------
            df_maintenance_saisi = st.session_state.get("maintenance_df")
            df_mnt_mois = (
                tableau_maintenance.calculerParMois(df_maintenance_saisi)
                if df_maintenance_saisi is not None
                else pd.DataFrame()
            )
            maintenance_disponible = (
                not df_mnt_mois.empty and df_mnt_mois["Planifiees"].sum() > 0
            )

            onglet_vue_ensemble, onglet_croisee, onglet_maintenance = st.tabs(
                ["Vue d'ensemble", "Analyse croisée", "Suivi des maintenances"]
            )

            # Colonnes catégorielles "pertinentes" (ni identifiants quasi
            # uniques, ni colonnes constantes), détectées automatiquement,
            # quel que soit leur nom. Limité à 3 colonnes maximum pour éviter
            # d'afficher une rangée de graphes peu lisibles.
            colonnes_categorielles = eda.colonnes_categorielles_pertinentes(
                pipeline.jeuDonnees.dataframe
            )[:3]
            cols_num_dispo = eda.colonnes_numeriques_triees_par_pertinence(
                pipeline.jeuDonnees.dataframe
            )

            pareto_num = None
            pareto_cat = None
            if "Machine" in pipeline.jeuDonnees.dataframe.columns:
                pareto_cat = "Machine"
                pareto_num = eda.colonne_temps_arret(pipeline.jeuDonnees.dataframe)
            else:
                eligible_cats = [
                    col
                    for col in colonnes_categorielles
                    if not AnalyseEDA._colonne_semble_date(pipeline.jeuDonnees.dataframe[col], col)
                ]
                pareto_cat = eligible_cats[0] if eligible_cats else None
                pareto_num = eda.colonne_temps_arret(pipeline.jeuDonnees.dataframe)

            # ---- Onglet 1 : Vue d'ensemble (toutes les analyses générales) ----
            with onglet_vue_ensemble:
                stats_df = pd.DataFrame([s.__dict__ for s in eda.statistiques])
                if not stats_df.empty:
                    st.subheader("Statistiques descriptives")
                    st.dataframe(stats_df, use_container_width=True)

                # Les valeurs manquantes n'ont pas de sens sur une saisie
                # manuelle (peu de lignes, structure fixe imposée par le
                # tableau de saisie) : on masque cette section dans ce cas.
                if not pipeline.estSaisieManuelle():
                    st.subheader("Valeurs manquantes par colonne")
                    tableau_manquantes = eda.tableauValeursManquantes(pipeline.jeuDonnees)
                    st.dataframe(
                        tableau_manquantes.rename(
                            columns={
                                "colonne": "Colonne",
                                "nb_valeurs_manquantes": "Nb manquantes",
                                "pourcentage_manquant": "% manquant",
                            }
                        ),
                        hide_index=True,
                        use_container_width=True,
                    )

                    st.subheader("Diagramme de Pareto")
                    if pareto_num and pareto_cat:
                        st.caption(
                            f"Classement ABC des catégories de « {pareto_cat} » "
                            f"selon leur poids dans le cumul de « {pareto_num} » : "
                            "classe A jusqu'à 80 % du cumul, classe B jusqu'à 95 %, "
                            "classe C au-delà."
                        )
                        fig_pareto_global = eda.afficherPareto(
                            pipeline.jeuDonnees, pareto_num, pareto_cat
                        )
                        if fig_pareto_global is not None:
                            st.pyplot(fig_pareto_global)
                        else:
                            st.caption("Pas assez de données pour tracer un diagramme de Pareto.")
                    else:
                        st.caption(
                            "Pas assez de colonnes numériques et catégorielles pour un "
                            "diagramme de Pareto."
                        )
                else:
                    st.caption(
                        "Visualisations générales (histogramme, Pareto) non affichées pour "
                        "les données saisies manuellement de maintenance planifiée/réalisée."
                    )

            # ---- Onglet 2 : Analyse croisée (numérique x catégoriel) -------
            with onglet_croisee:
                if cols_num_dispo and colonnes_categorielles:
                    ac1, ac2 = st.columns(2)
                    with ac1:
                        colonne_num_choisie = st.selectbox(
                            "Colonne numérique", cols_num_dispo, key="eda_colonne_num"
                        )
                    with ac2:
                        colonne_cat_choisie = st.selectbox(
                            "Regrouper par", colonnes_categorielles, key="eda_colonne_cat"
                        )
                    fig_top = eda.afficherTopCategories(
                        pipeline.jeuDonnees, colonne_num_choisie, colonne_cat_choisie
                    )
                    if fig_top is not None:
                        st.pyplot(fig_top)
                    kpi = eda.calculerAgregations(
                        pipeline.jeuDonnees, colonne_num_choisie, [colonne_cat_choisie]
                    )
                    for nom_kpi, df_kpi in kpi.items():
                        st.caption(f"Détail par {nom_kpi}")
                        st.dataframe(df_kpi, use_container_width=True)
                else:
                    st.caption("Pas assez de colonnes numériques et catégorielles pour une analyse croisée.")

            # ---- Onglet 3 : Suivi des maintenances --------------------------
            with onglet_maintenance:
                if maintenance_disponible:
                    kpi_mnt = tableau_maintenance.calculerKPI(df_mnt_mois)

                    k1, k2, k3, k4, k5 = st.columns(5)
                    k1.metric("Total planifié", kpi_mnt["total_planifie"])
                    k2.metric("Total réalisé", kpi_mnt["total_realise"])
                    k3.metric("Taux global", f"{kpi_mnt['taux_global']} %")
                    k4.metric("Mois dans l'objectif", kpi_mnt["mois_objectif_atteint"])
                    k5.metric("Mois sous l'objectif", kpi_mnt["mois_sous_objectif"])

                    with st.expander("Voir les graphiques détaillés", expanded=False):
                        gm1, gm2 = st.columns(2)
                        with gm1:
                            st.pyplot(tableau_maintenance.graphiquePlanifieRealise(df_mnt_mois))
                        with gm2:
                            st.pyplot(tableau_maintenance.graphiqueEvolutionTaux(df_mnt_mois))

                        gm3, gm4 = st.columns(2)
                        with gm3:
                            st.pyplot(tableau_maintenance.graphiquePerformanceMensuelle(df_mnt_mois))
                        with gm4:
                            fig_repartition = tableau_maintenance.graphiqueRepartitionAnnuelle(kpi_mnt)
                            if fig_repartition is not None:
                                st.pyplot(fig_repartition)

                        st.subheader("Diagramme de Pareto — mois critiques")
                        fig_pareto = tableau_maintenance.graphiquePareto(df_mnt_mois)
                        if fig_pareto is not None:
                            st.pyplot(fig_pareto)
                        else:
                            st.caption("Aucun écart de maintenance à afficher (objectifs atteints partout).")

                    st.subheader("Analyse automatique")
                    st.markdown(tableau_maintenance.genererResume(df_mnt_mois, kpi_mnt))

                    st.download_button(
                        "Exporter le suivi des maintenances (Excel)",
                        data=tableau_maintenance.exporterExcel(
                            df_maintenance_saisi, df_mnt_mois, kpi_mnt
                        ),
                        file_name="Suivi_Maintenances.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True,
                    )
                else:
                    st.caption(
                        "Aucune donnée de maintenance saisie pour le moment. "
                        "Renseigne le tableau depuis l'étape Import du fichier → "
                        "Saisie manuelle."
                    )

            st.caption("Passe à l'étape Prétraitement dans le menu à gauche.")


# --------------------------------------------------------------------------
# 3. PRETRAITEMENT
# --------------------------------------------------------------------------
elif page_num == 3:
    if etape_terminee < 2:
        st.warning("Réalise d'abord l'étape Analyse exploratoire (EDA).")
    elif pipeline.estSaisieManuelle():
        st.warning(
            "Le prétraitement n'est pas disponible pour les données de saisie "
            "manuelle de suivi de maintenance."
        )
        st.caption(
            "Utilise l'onglet 'Suivi des maintenances' pour analyser et exporter "
            "tes données, ou recharge un fichier de données structurées pour "
            "accéder au Machine Learning."
        )
    else:
        p1, p2 = st.columns(2)
        with p1:
            methode_norm = st.selectbox("Méthode de normalisation", ["standard", "minmax"], index=0)
        with p2:
            methode_enc = st.selectbox(
                "Méthode d'encodage des variables catégorielles", ["onehot", "label"], index=0
            )

        if st.button("Lancer le prétraitement", type="primary"):
            pipeline.pretraitement.methodeNormalisation = methode_norm
            pipeline.pretraitement.methodeEncodage = methode_enc
            pipeline.pretraiterDonnees()
            st.session_state["etape_terminee"] = max(st.session_state["etape_terminee"], 3)

        if st.session_state["etape_terminee"] >= 3:
            st.info(
                f"Doublons supprimés : {pipeline.pretraitement.rapport['doublons_supprimes']} — "
                f"Valeurs imputées : {pipeline.pretraitement.rapport['valeurs_imputees']}"
            )
            if pipeline.source_cible == "regles_generiques":
                rr = pipeline.rapport_regles
                st.caption(
                    f"Aucune colonne « {COLONNE_CIBLE} » trouvée dans le fichier source : "
                    "les étiquettes Utile/Non_utile ont été générées automatiquement à "
                    "partir de règles génériques appliquées à N'IMPORTE QUEL fichier — "
                    f"lignes trop vides : {rr.get('trop_vide', 0)}, "
                    f"doublons : {rr.get('doublon', 0)}, "
                    f"valeurs aberrantes : {rr.get('valeur_aberrante', 0)} "
                    f"(→ {rr.get('non_utile_total', 0)} lignes Non_utile / "
                    f"{rr.get('utile_total', 0)} lignes Utile)."
                )
            else:
                st.caption(
                    f"Colonne « {COLONNE_CIBLE} » trouvée dans le fichier source : "
                    "elle a été utilisée comme vérité terrain plutôt que les règles automatiques."
                )
            st.subheader("Aperçu des données prétraitées")
            st.dataframe(pipeline._jeu_pretraite.dataframe.head(50), use_container_width=True)
            st.caption("Passe à l'étape Machine Learning dans le menu à gauche.")


# --------------------------------------------------------------------------
# 4. MACHINE LEARNING (RandomForestClassifier)
# --------------------------------------------------------------------------
elif page_num == 4:
    if etape_terminee < 3:
        st.warning("Réalise d'abord l'étape Prétraitement.")
    elif pipeline.estSaisieManuelle():
        st.warning(
            "Machine Learning n'est pas disponible pour les données de saisie "
            "manuelle de suivi de maintenance."
        )
        st.caption(
            "Recharge un fichier structuré ou passe par l'onglet 'Suivi des "
            "maintenances' pour analyser tes données manuellement."
        )
    else:
        modele_choisi = st.selectbox(
            "Modèle de Machine Learning",
            ["Random Forest", "Régression Logistique", "Gradient Boosting", "SVM (Support Vector Machine)"],
            key="ml_modele_choisi",
        )

        m1, m2, m3 = st.columns(3)

        params = {}
        if modele_choisi == "Random Forest":
            with m1:
                params["nombreArbres"] = st.slider("Nombre d'arbres", 50, 500, 200, step=50, key="ml_rf_nb_arbres")
            with m2:
                profondeur = st.slider("Profondeur max (0 = illimitée)", 0, 30, 0, key="ml_rf_profondeur")
                params["profondeurMax"] = None if profondeur == 0 else profondeur

        elif modele_choisi == "Régression Logistique":
            with m1:
                params["C"] = st.slider("Régularisation (C)", 0.01, 10.0, 1.0, step=0.01, key="ml_lr_c")
            with m2:
                params["maxIter"] = st.slider("Itérations max", 100, 2000, 500, step=100, key="ml_lr_maxiter")

        elif modele_choisi == "Gradient Boosting":
            with m1:
                params["nombreArbres"] = st.slider("Nombre d'estimateurs", 50, 500, 150, step=50, key="ml_gb_nb_arbres")
            with m2:
                params["tauxApprentissage"] = st.slider("Taux d'apprentissage", 0.01, 0.5, 0.1, step=0.01, key="ml_gb_taux")
            with m3:
                params["profondeurMax"] = st.slider("Profondeur max", 1, 10, 3, key="ml_gb_profondeur")

        elif modele_choisi == "SVM (Support Vector Machine)":
            with m1:
                params["C"] = st.slider("Régularisation (C)", 0.01, 10.0, 1.0, step=0.01, key="ml_svm_c")
            with m2:
                params["kernel"] = st.selectbox("Noyau", ["rbf", "linear", "poly"], key="ml_svm_kernel")

        with m3 if modele_choisi == "Random Forest" else m1:
            taille_test = st.slider("Taille du jeu de test", 0.1, 0.4, 0.25, step=0.05, key="ml_taille_test")

        if st.button("Entraîner le modèle et prédire", type="primary", key="ml_bouton_entrainer"):
            try:
                rapport = pipeline.entrainerModele(
                    nomModele=modele_choisi,
                    parametres=params,
                    taille_test=taille_test,
                )
                df_resultat = pipeline.predire()
                st.session_state["df_resultat"] = df_resultat
                st.session_state["etape_terminee"] = max(st.session_state["etape_terminee"], 4)
            except ValueError as e:
                st.error(str(e))

        if st.session_state["etape_terminee"] >= 4:
            rapport = pipeline.rapport
            r1, r2, r3, r4 = st.columns(4)
            r1.metric("Accuracy", f"{rapport.accuracy * 100:.1f} %")
            r2.metric("Precision", f"{rapport.precision * 100:.1f} %")
            r3.metric("Recall", f"{rapport.recall * 100:.1f} %")
            r4.metric("F1-score", f"{rapport.f1Score * 100:.1f} %")

            st.subheader("Matrice de confusion")
            st.dataframe(
                matrice_confusion_vers_dataframe(rapport.matriceConfusion),
                use_container_width=True,
            )

# --------------------------------------------------------------------------
# 5. EXPORT DU RESULTAT
# --------------------------------------------------------------------------
elif page_num == 5:
    if etape_terminee < 4:
        st.warning("Réalise d'abord l'étape Machine Learning.")
    else:
        df_resultat = st.session_state["df_resultat"]
        par_classe = pipeline.exporterResultat(df_resultat)
        df_utile = par_classe.get("Utile", df_resultat.iloc[0:0])
        df_non_utile = par_classe.get("Non_utile", df_resultat.iloc[0:0])
        st.session_state["etape_terminee"] = max(st.session_state["etape_terminee"], 5)

        s1, s2, s3 = st.columns(3)
        s1.metric("Lignes totales", len(df_resultat))
        s2.metric("Utile", len(df_utile))
        s3.metric("Non_utile", len(df_non_utile))

        tabs = st.tabs([f"{classe} ({len(df)})" for classe, df in par_classe.items()])
        for tab, (classe, df) in zip(tabs, par_classe.items()):
            with tab:
                st.dataframe(df, use_container_width=True)

        st.markdown("### Export")
        st.download_button(
            f"Télécharger le fichier nettoyé ({DEFAULT_OUTPUT_NAME})",
            data=pipeline.exportExcel.genererExportComplet(df_utile, df_non_utile),
            file_name=DEFAULT_OUTPUT_NAME,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
        st.caption("Le fichier Excel reste sous sa forme normale : lignes utiles nettoyées.")

        inclure_pdf = st.checkbox("Générer aussi un PDF d'analyse détaillée", value=True)

        if inclure_pdf:
            eda = pipeline.analyseEDA
            pareto_num = eda.colonne_temps_arret(pipeline.jeuDonnees.dataframe)
            if "Machine" in pipeline.jeuDonnees.dataframe.columns:
                pareto_cat = "Machine"
            else:
                eligible_cats = [
                    col
                    for col in eda.colonnes_categorielles_pertinentes(pipeline.jeuDonnees.dataframe)
                    if not AnalyseEDA._colonne_semble_date(pipeline.jeuDonnees.dataframe[col], col)
                ]
                pareto_cat = eligible_cats[0] if eligible_cats else None

            analyses = {}
            figures = {}

            try:
                stats_df = pd.DataFrame([s.__dict__ for s in eda.statistiques])
                if not stats_df.empty:
                    analyses["Statistiques descriptives"] = stats_df
            except Exception:
                pass

            try:
                mv = eda.tableauValeursManquantes(pipeline.jeuDonnees)
                if mv is not None and not mv.empty:
                    analyses["Valeurs manquantes"] = mv
            except Exception:
                pass

            try:
                if pareto_num and pareto_cat:
                    fig = eda.afficherPareto(pipeline.jeuDonnees, pareto_num, pareto_cat)
                    if fig is not None:
                        figures[f"Pareto - {pareto_cat} par {pareto_num}"] = fig
            except Exception:
                pass

            try:
                fig_hist = eda.afficherHistogrammes(pipeline.jeuDonnees, max_colonnes=2)
                if fig_hist is not None:
                    figures["Histograms"] = fig_hist
            except Exception:
                pass

            try:
                if pareto_cat and pareto_num:
                    fig_top = eda.afficherTopCategories(pipeline.jeuDonnees, pareto_num, pareto_cat, top_n=10)
                    if fig_top is not None:
                        figures[f"Top catégories {pareto_cat} par {pareto_num}"] = fig_top
            except Exception:
                pass

            try:
                aggs = eda.calculerAgregations(
                    pipeline.jeuDonnees,
                    colonne_numerique=pareto_num,
                    colonnes_groupe=[pareto_cat] if pareto_cat else None,
                )
                for nom, df_ag in aggs.items():
                    if df_ag is not None and not df_ag.empty:
                        analyses[f"Aggrégation par {nom}"] = df_ag
            except Exception:
                pass

            meta = {
                "Nb lignes totales": str(len(df_resultat)),
                "Nb utiles": str(len(df_utile)),
                "Nb non utiles": str(len(df_non_utile)),
                "Fichier source": pipeline.fichierExcel.nom if pipeline.fichierExcel else "Saisie manuelle",
            }
            pdf_bytes = pipeline.exportExcel.genererRapportPDF(
                analyses=analyses,
                figures=figures,
                meta=meta,
                titre="Analyse détaillée du fichier nettoyé",
            )

            st.download_button(
                "Télécharger le rapport PDF d'analyse détaillée",
                data=pdf_bytes,
                file_name="Rapport_Analyse_Detailee.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
            st.caption("Le PDF contient une analyse détaillée du fichier nettoyé et des graphiques.")


# --------------------------------------------------------------------------
# Pied de page
# --------------------------------------------------------------------------
st.markdown(
    f"""
    <hr style="border: none; border-top: 1px solid {TE_ORANGE_LIGHT}; margin-top: 40px;">
    
    """,
    unsafe_allow_html=True,
)
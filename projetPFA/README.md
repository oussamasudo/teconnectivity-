# Projet PFA — Nettoyage intelligent de données par Machine Learning

Pipeline complet **générique** : **Import Excel/CSV → Analyse exploratoire
(EDA) → Prétraitement → Machine Learning → Export Excel → Power BI**,
conforme à la fiche technique du projet.

L'outil s'adapte à **n'importe quel fichier Excel/CSV** : aucune colonne,
aucune feuille, aucune structure n'est supposée connue à l'avance.

## Objectif

À partir de n'importe quel fichier de données tabulaires, le système :
1. Importe et valide le fichier source (détection automatique de la ligne
   d'en-tête réelle, même sur un export "brut" avec logo/titres).
2. Explore les données (statistiques, corrélations, visualisations,
   analyse croisée avec détection automatique des colonnes pertinentes).
3. Nettoie les données (doublons, valeurs manquantes, encodage, scaling).
4. Génère une cible **Utile / Non_utile** par des règles génériques
   (ligne trop incomplète, dupliquée, valeur aberrante) si le fichier ne
   fournit pas déjà de colonne cible, puis entraîne un classifieur
   (Random Forest / Régression Logistique / Gradient Boosting / SVM) qui
   apprend à généraliser ces règles.
5. Exporte un fichier Excel propre (`Donnees_Nettoyees.xlsx`, 2 onglets :
   lignes utiles avec colonnes pertinentes uniquement, et lignes
   rejetées pour l'audit).
6. Ce fichier alimente ensuite un tableau de bord **Power BI** (hors
   code, étape externe).

## Structure du projet

```
projetPFA/
├── app.py                     # Interface Streamlit
├── config.py                  # Configuration centrale (règles, seuils)
├── requirements.txt
├── test_pipeline.py           # Script de test bout-en-bout (CLI, sans Streamlit)
├── models/
│   ├── fichier_excel.py       # Bloc 1 - Import générique (détection d'en-tête auto)
│   ├── jeu_donnees.py         # Classe JeuDonnees
│   ├── analyse_eda.py         # Bloc 2 - EDA générique
│   ├── pretraitement.py       # Bloc 3 - Prétraitement
│   ├── modele_ml.py           # Bloc 4 - ML + règles génériques Utile/Non_utile
│   ├── rapport_evaluation.py  # RapportEvaluation (accuracy, precision, recall, F1)
│   ├── export_excel.py        # Bloc 5 - Export générique (2 onglets)
│   ├── pipeline_principal.py  # Orchestrateur (PipelinePrincipal)
│   └── logger.py
├── utils/
│   ├── file_utils.py
│   ├── excel_utils.py
│   └── metrics.py
├── data/                      # Fichiers d'exemple pour tester le pipeline
├── outputs/                   # Fichiers générés (Donnees_Nettoyees.xlsx, audit_report.xlsx)
└── logs/app.log
```

## Installation

```bash
python -m venv venv
source venv/bin/activate        # ou venv\Scripts\activate sous Windows
pip install -r requirements.txt
```

## Lancement

```bash
streamlit run app.py
```

Puis, dans l'interface :
1. Uploader n'importe quel fichier Excel/CSV (ou coller une URL).
2. (Facultatif) Indiquer les colonnes que TU juges obligatoires pour CE
   fichier — l'outil avertit sans bloquer si elles sont absentes.
3. Cliquer sur **Importer** → **Lancer l'analyse exploratoire** →
   **Lancer le prétraitement** → **Entraîner le modèle et prédire**.
4. Télécharger `Donnees_Nettoyees.xlsx` (lignes utiles + audit des
   lignes rejetées).
5. Brancher `Donnees_Nettoyees.xlsx` comme source de données dans Power BI.

## Tester sans l'interface (CLI)

```bash
python test_pipeline.py
```

Exécute le pipeline complet sur les fichiers d'exemple du dossier `data/`
(y compris un fichier volontairement "désorganisé", pour valider la
détection automatique de l'en-tête) et affiche un résumé des métriques.

## Note sur l'apprentissage supervisé

Les modèles ont besoin d'une colonne cible (`Utile` : 0/1, ou "Utile"/
"Non_utile") pour s'entraîner. Si le fichier source ne contient pas cette
colonne, elle est **générée automatiquement** par des règles génériques
(taux de valeurs manquantes de la ligne, doublons, valeurs numériques
aberrantes détectées par la méthode IQR) dans
`models/modele_ml.py::generer_labels_utilite`. Le modèle apprend ensuite à
généraliser ces règles à partir des autres caractéristiques de chaque
ligne. Si tu disposes d'un jeu déjà étiqueté, ajoute simplement une
colonne `Utile` (0/1) à ton fichier source : elle sera utilisée en
priorité comme vérité terrain, à la place des règles automatiques.

## Environnement technique

- Python, pandas, numpy
- scikit-learn (RandomForest / Régression Logistique / Gradient Boosting /
  SVM, StandardScaler/MinMaxScaler)
- matplotlib, seaborn (visualisations EDA)
- openpyxl (lecture/écriture Excel)
- Streamlit (interface utilisateur)
- Power BI (restitution finale, hors code)

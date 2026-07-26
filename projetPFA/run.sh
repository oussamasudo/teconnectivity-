#!/bin/bash
# ---------------------------------------------------------------------------
# run.sh — Lance tout le projet en une seule commande (Mac / Linux).
#
# Utilisation :
#   chmod +x run.sh      (une seule fois, pour rendre le script exécutable)
#   ./run.sh
# ---------------------------------------------------------------------------

set -e  # arrête le script si une commande échoue

echo "==> Vérification de l'environnement virtuel..."
if [ ! -d "venv" ]; then
    echo "==> Création de l'environnement virtuel (venv)..."
    python3 -m venv venv
fi

echo "==> Activation de l'environnement virtuel..."
source venv/bin/activate

echo "==> Installation des dépendances (requirements.txt)..."
pip install -q --upgrade pip
pip install -q -r requirements.txt

echo "==> Lancement de l'application Streamlit..."
streamlit run app.py

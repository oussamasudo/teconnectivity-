@echo off
REM ---------------------------------------------------------------------
REM run.bat — Lance tout le projet en une seule commande (Windows).
REM
REM Utilisation : double-clique sur run.bat, ou dans un terminal : run.bat
REM ---------------------------------------------------------------------

echo ==^> Verification de l'environnement virtuel...
if not exist "venv\" (
    echo ==^> Creation de l'environnement virtuel ^(venv^)...
    python -m venv venv
)

echo ==^> Activation de l'environnement virtuel...
call venv\Scripts\activate.bat

echo ==^> Installation des dependances ^(requirements.txt^)...
pip install -q --upgrade pip
pip install -q -r requirements.txt

echo ==^> Lancement de l'application Streamlit...
streamlit run app.py

pause

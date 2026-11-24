# Atelier 2 : Modularisation du code (Churn Modelling)

## Installation
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

## Utilisation
python main.py --prepare             # charger et prétraiter les données
python main.py --train               # entraîner et sauvegarder le modèle
python main.py --evaluate            # charger et évaluer le modèle
python main.py --train --evaluate    # pipeline complet

## Fichiers
- model_pipeline.py : fonctions prepare_data, train_model, evaluate_model, save_model, load_model
- main.py : exécution du pipeline via des arguments CLI

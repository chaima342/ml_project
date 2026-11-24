"""
main.py
Point d'entrée du pipeline ML : exécute les étapes via des arguments CLI.

Exemples :
    python main.py --prepare
    python main.py --train
    python main.py --evaluate
    python main.py --train --evaluate
"""

import argparse

from model_pipeline import (
    prepare_data,
    train_model,
    evaluate_model,
    save_model,
    load_model,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Pipeline ML - Churn Modelling")
    parser.add_argument("--prepare", action="store_true",
                        help="Charger et prétraiter les données")
    parser.add_argument("--train", action="store_true",
                        help="Entraîner le modèle et le sauvegarder")
    parser.add_argument("--evaluate", action="store_true",
                        help="Charger le modèle sauvegardé et l'évaluer")
    parser.add_argument("--data", default="Churn_Modelling.csv",
                        help="Chemin du fichier CSV (défaut : Churn_Modelling.csv)")
    parser.add_argument("--model", default="classifier.joblib",
                        help="Chemin du modèle (défaut : classifier.joblib)")
    return parser.parse_args()


def main():
    args = parse_args()

    if not (args.prepare or args.train or args.evaluate):
        print("Aucune action choisie. Utilisez --prepare, --train ou --evaluate.")
        print("Aide : python main.py --help")
        return

    # Préparation des données (nécessaire pour toutes les étapes)
    X_train, X_test, y_train, y_test = prepare_data(args.data)

    if args.train:
        model = train_model(X_train, y_train)
        save_model(model, args.model)

    if args.evaluate:
        model = load_model(args.model)
        evaluate_model(model, X_test, y_test)


if __name__ == "__main__":
    main()

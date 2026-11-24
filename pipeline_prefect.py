"""
pipeline_prefect.py
Orchestration du pipeline ML avec Prefect.

Exemples :
    python pipeline_prefect.py --flow all
    python pipeline_prefect.py --flow train
    python pipeline_prefect.py --flow evaluate
"""

import argparse
import subprocess
import sys

from prefect import flow, task

from model_pipeline import (
    prepare_data,
    train_model,
    evaluate_model,
    save_model,
    load_model,
)

DATA_PATH = "Churn_Modelling.csv"
MODEL_PATH = "classifier.joblib"


# ---------- TASKS ----------

@task(name="Installer les dependances")
def install_dependencies():
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
        check=True,
    )


@task(name="Preparation des donnees")
def prepare_task(csv_path=DATA_PATH):
    return prepare_data(csv_path)


@task(name="Entrainement du modele")
def train_task(X_train, y_train):
    return train_model(X_train, y_train)


@task(name="Sauvegarder le modele")
def save_task(model, path=MODEL_PATH):
    save_model(model, path)


@task(name="Charger le modele")
def load_task(path=MODEL_PATH):
    return load_model(path)


@task(name="Evaluer le modele")
def evaluate_task(model, X_test, y_test):
    results = evaluate_model(model, X_test, y_test)
    return results["accuracy"]


# ---------- FLOWS ----------

@flow(name="train")
def train_flow():
    X_train, X_test, y_train, y_test = prepare_task()
    model = train_task(X_train, y_train)
    save_task(model)


@flow(name="evaluate")
def evaluate_flow():
    X_train, X_test, y_train, y_test = prepare_task()
    model = load_task()
    accuracy = evaluate_task(model, X_test, y_test)
    print(f"Accuracy finale : {accuracy * 100:.2f} %")


@flow(name="all")
def all_flow():
    # Etapes 2-3-4 et 6 : prepare, train, save, evaluate
    X_train, X_test, y_train, y_test = prepare_task()
    model = train_task(X_train, y_train)
    save_task(model)
    evaluate_task(model, X_test, y_test)


# ---------- CLI ----------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline ML avec Prefect")
    parser.add_argument(
        "--flow",
        choices=["all", "train", "entrainement", "evaluate"],
        default="all",
        help="Flow à lancer",
    )
    args = parser.parse_args()

    if args.flow == "all":
        all_flow()
    elif args.flow in ("train", "entrainement"):
        train_flow()
    elif args.flow == "evaluate":
        evaluate_flow()

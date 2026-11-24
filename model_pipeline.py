"""
model_pipeline.py
Fonctions modulaires du pipeline ML (prédiction du churn client).
"""

import os

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def prepare_data(csv_path="Churn_Modelling.csv", test_size=0.2, random_state=1):
    """
    Charge et prétraite les données.

    Args:
        csv_path (str): chemin du fichier CSV.
        test_size (float): proportion du jeu de test (entre 0 et 1).
        random_state (int): graine aléatoire pour le split.

    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Fichier introuvable : {csv_path}")

    df = pd.read_csv(csv_path)

    colonnes_requises = {"Gender", "Surname", "Geography", "Exited",
                         "RowNumber", "CustomerId"}
    manquantes = colonnes_requises - set(df.columns)
    if manquantes:
        raise ValueError(f"Colonnes manquantes dans le CSV : {manquantes}")

    # Encodage de Gender
    df["Gender"] = LabelEncoder().fit_transform(df["Gender"])

    # Suppression des colonnes inutiles
    df = df.drop(columns=["Surname", "Geography"])

    # Séparation features / cible
    y = df["Exited"]
    X = df.drop(columns=["Exited", "RowNumber", "CustomerId"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"[prepare_data] Train : {X_train.shape} | Test : {X_test.shape}")
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train, n_estimators=100, random_state=42):
    """
    Entraîne un RandomForestClassifier.

    Args:
        X_train: features d'entraînement.
        y_train: cible d'entraînement.
        n_estimators (int): nombre d'arbres.
        random_state (int): graine aléatoire.

    Returns:
        RandomForestClassifier: modèle entraîné.
    """
    if len(X_train) != len(y_train):
        raise ValueError("X_train et y_train n'ont pas la même taille.")

    model = RandomForestClassifier(n_estimators=n_estimators,
                                   random_state=random_state)
    model.fit(X_train, y_train)
    print("[train_model] Modèle entraîné.")
    return model


def evaluate_model(model, X_test, y_test):
    """
    Évalue le modèle sur le jeu de test.

    Args:
        model: modèle entraîné.
        X_test: features de test.
        y_test: cible de test.

    Returns:
        dict: accuracy, rapport de classification, matrice de confusion.
    """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    matrix = confusion_matrix(y_test, y_pred)

    print(f"[evaluate_model] Accuracy : {accuracy * 100:.2f} %")
    print(report)
    print("Matrice de confusion :")
    print(matrix)

    return {"accuracy": accuracy, "report": report, "confusion_matrix": matrix}


def save_model(model, path="classifier.joblib"):
    """
    Sauvegarde le modèle avec joblib.

    Args:
        model: modèle à sauvegarder.
        path (str): chemin du fichier de sortie.
    """
    joblib.dump(model, path)
    print(f"[save_model] Modèle sauvegardé : {path}")


def load_model(path="classifier.joblib"):
    """
    Charge un modèle sauvegardé avec joblib.

    Args:
        path (str): chemin du fichier du modèle.

    Returns:
        Le modèle chargé.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Modèle introuvable : {path}")
    model = joblib.load(path)
    print(f"[load_model] Modèle chargé : {path}")
    return model

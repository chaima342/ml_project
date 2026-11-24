"""Tests individuels des fonctions de model_pipeline.py"""

import os
import tempfile

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

import model_pipeline as mp


def raises(exc, func, *args, **kwargs):
    """Retourne True si func(*args) lève bien l'exception attendue."""
    try:
        func(*args, **kwargs)
    except exc:
        return True
    return False


def test_prepare_data():
    X_train, X_test, y_train, y_test = mp.prepare_data("Churn_Modelling.csv")
    assert X_train.shape == (8000, 9)
    assert X_test.shape == (2000, 9)
    assert len(y_train) == 8000 and len(y_test) == 2000
    assert X_train.isna().sum().sum() == 0
    for col in ["Exited", "RowNumber", "CustomerId", "Surname", "Geography"]:
        assert col not in X_train.columns
    assert set(y_train.unique()) <= {0, 1}

    # Cas d'erreur 1 : fichier absent
    assert raises(FileNotFoundError, mp.prepare_data, "inexistant.csv")

    # Cas d'erreur 2 : colonnes manquantes
    with tempfile.TemporaryDirectory() as tmp:
        bad = os.path.join(tmp, "bad.csv")
        pd.DataFrame({"a": [1, 2], "b": [3, 4]}).to_csv(bad, index=False)
        assert raises(ValueError, mp.prepare_data, bad)
    print("OK  prepare_data")
    return X_train, X_test, y_train, y_test


def test_train_model(X_train, y_train):
    model = mp.train_model(X_train, y_train)
    assert isinstance(model, RandomForestClassifier)
    assert hasattr(model, "predict")
    # Cas d'erreur : tailles différentes
    assert raises(ValueError, mp.train_model, X_train.iloc[:10], y_train)
    print("OK  train_model")
    return model


def test_evaluate_model(model, X_test, y_test):
    result = mp.evaluate_model(model, X_test, y_test)
    assert set(result) == {"accuracy", "report", "confusion_matrix"}
    assert 0 <= result["accuracy"] <= 1
    assert result["confusion_matrix"].shape == (2, 2)
    print("OK  evaluate_model")


def test_save_and_load(model, X_test):
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "m.joblib")
        mp.save_model(model, path)
        assert os.path.exists(path)
        loaded = mp.load_model(path)
        # Le modèle rechargé doit donner les mêmes prédictions
        assert (loaded.predict(X_test) == model.predict(X_test)).all()
    # Cas d'erreur : modèle absent
    assert raises(FileNotFoundError, mp.load_model, "inexistant.joblib")
    print("OK  save_model / load_model")


def test_docstrings():
    for f in [mp.prepare_data, mp.train_model, mp.evaluate_model,
              mp.save_model, mp.load_model]:
        assert f.__doc__ and len(f.__doc__.strip()) > 20, f.__name__
    print("OK  docstrings")


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = test_prepare_data()
    model = test_train_model(X_train, y_train)
    test_evaluate_model(model, X_test, y_test)
    test_save_and_load(model, X_test)
    test_docstrings()
    print("\nTous les tests sont passés.")

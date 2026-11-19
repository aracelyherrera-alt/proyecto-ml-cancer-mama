"""
Logica de Machine Learning del proyecto.
Aqui se carga el dataset, se arma el pipeline, se entrena y se calculan las metricas.
"""
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# en versiones nuevas de sklearn probability=True avisa que se va a cambiar
warnings.filterwarnings("ignore", category=FutureWarning)

RAIZ = Path(__file__).resolve().parent.parent
RUTA_DATOS = RAIZ / "data" / "wdbc.csv"
RUTA_MODELO = RAIZ / "modelos" / "modelo.joblib"

MODELOS_DISPONIBLES = ["random_forest", "regresion_logistica", "svm", "knn"]

# parametros que acepta cada modelo y su valor por defecto
PARAMS_POR_DEFECTO = {
    "random_forest": {"n_estimators": 100, "max_depth": None},
    "regresion_logistica": {"C": 1.0, "max_iter": 1000},
    "svm": {"C": 1.0, "kernel": "rbf"},
    "knn": {"n_neighbors": 5},
}


def cargar_datos():
    """Lee el csv y separa X (30 variables) de y (1 = maligno, 0 = benigno)."""
    df = pd.read_csv(RUTA_DATOS)
    y = (df["diagnosis"] == "M").astype(int)
    X = df.drop(columns="diagnosis")
    return X, y


def crear_modelo(nombre, params=None, semilla=42):
    p = {**PARAMS_POR_DEFECTO[nombre], **(params or {})}
    if nombre == "random_forest":
        clf = RandomForestClassifier(
            n_estimators=int(p["n_estimators"]),
            max_depth=None if p["max_depth"] in (None, 0) else int(p["max_depth"]),
            random_state=semilla,
        )
    elif nombre == "regresion_logistica":
        clf = LogisticRegression(C=float(p["C"]), max_iter=int(p["max_iter"]))
    elif nombre == "svm":
        clf = SVC(C=float(p["C"]), kernel=p["kernel"], probability=True,
                  random_state=semilla)
    elif nombre == "knn":
        clf = KNeighborsClassifier(n_neighbors=int(p["n_neighbors"]))
    else:
        raise ValueError(f"Modelo no valido: {nombre}")
    # el escalado es importante para SVM, KNN y regresion logistica
    return Pipeline([("escalador", StandardScaler()), ("clf", clf)]), p


def entrenar(nombre="random_forest", params=None, test_size=0.2, semilla=42,
             guardar=True):
    """Entrena el modelo y devuelve un diccionario con las metricas."""
    X, y = cargar_datos()
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=semilla, stratify=y
    )
    modelo, p = crear_modelo(nombre, params, semilla)
    modelo.fit(X_tr, y_tr)

    pred = modelo.predict(X_te)
    prob = modelo.predict_proba(X_te)[:, 1]
    cm = confusion_matrix(y_te, pred)
    fpr, tpr, _ = roc_curve(y_te, prob)
    cv = cross_val_score(modelo, X, y, cv=5, scoring="accuracy")

    metricas = {
        "modelo": nombre,
        "parametros": p,
        "test_size": test_size,
        "n_train": int(len(X_tr)),
        "n_test": int(len(X_te)),
        "accuracy": round(float(accuracy_score(y_te, pred)), 4),
        "precision": round(float(precision_score(y_te, pred)), 4),
        "recall": round(float(recall_score(y_te, pred)), 4),
        "f1": round(float(f1_score(y_te, pred)), 4),
        "roc_auc": round(float(roc_auc_score(y_te, prob)), 4),
        "cv_media": round(float(cv.mean()), 4),
        "cv_desv": round(float(cv.std()), 4),
        "matriz_confusion": cm.tolist(),
        "roc": {"fpr": np.round(fpr, 4).tolist(), "tpr": np.round(tpr, 4).tolist()},
    }

    # importancia de variables (solo si el modelo la tiene)
    clf = modelo.named_steps["clf"]
    if hasattr(clf, "feature_importances_"):
        imp = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        imp = np.abs(clf.coef_[0])
    else:
        imp = None
    if imp is not None:
        orden = np.argsort(imp)[::-1][:10]
        metricas["importancias"] = [
            {"variable": X.columns[i], "valor": round(float(imp[i]), 4)} for i in orden
        ]

    if guardar:
        RUTA_MODELO.parent.mkdir(exist_ok=True)
        joblib.dump({"modelo": modelo, "columnas": list(X.columns),
                     "metricas": metricas}, RUTA_MODELO)
    return metricas


def cargar_modelo_guardado():
    if not RUTA_MODELO.exists():
        return None
    return joblib.load(RUTA_MODELO)


def predecir(modelo, columnas, datos: dict):
    """Recibe un dict con las 30 variables y devuelve diagnostico y probabilidad."""
    fila = pd.DataFrame([[datos[c] for c in columnas]], columns=columnas)
    prob_maligno = float(modelo.predict_proba(fila)[0, 1])
    return {
        "diagnostico": "Maligno" if prob_maligno >= 0.5 else "Benigno",
        "probabilidad_maligno": round(prob_maligno, 4),
        "probabilidad_benigno": round(1 - prob_maligno, 4),
    }


if __name__ == "__main__":
    for m in MODELOS_DISPONIBLES:
        r = entrenar(m, guardar=False)
        print(f"{m:22s} acc={r['accuracy']} rec={r['recall']} f1={r['f1']} auc={r['roc_auc']} cv={r['cv_media']}")

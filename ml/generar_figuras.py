"""Corre los 4 modelos y guarda las figuras y la tabla de resultados (para el informe)."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ml import entrenamiento as ml

OUT = ml.RAIZ / "docs" / "figuras"
OUT.mkdir(parents=True, exist_ok=True)
NOMBRES = {"random_forest": "Random Forest", "regresion_logistica": "Reg. logistica",
           "svm": "SVM", "knn": "KNN"}
COL = {"random_forest": "#1b6ca8", "regresion_logistica": "#e07a1f", "svm": "#2a9d8f", "knn": "#8d6bb5"}
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})

res = {m: ml.entrenar(m, guardar=False) for m in ml.MODELOS_DISPONIBLES}
json.dump(res, open(OUT / "resultados.json", "w"), indent=1)

# 1. distribucion de clases
X, y = ml.cargar_datos()
fig, ax = plt.subplots(figsize=(4.5, 3.4))
c = y.value_counts().sort_index()
ax.bar(["Benigno", "Maligno"], c.values, color=["#2a9d8f", "#c0392b"])
for i, v in enumerate(c.values):
    ax.text(i, v + 5, f"{v} ({v/len(y):.0%})", ha="center")
ax.set_ylabel("Casos"); ax.set_title("Distribucion de las clases")
fig.tight_layout(); fig.savefig(OUT / "clases.png", dpi=200); plt.close()

# 2. comparacion de modelos
met = ["accuracy", "precision", "recall", "f1"]
fig, ax = plt.subplots(figsize=(7.5, 3.8))
w = 0.2
for i, m in enumerate(res):
    ax.bar(np.arange(4) + i * w, [res[m][k] for k in met], w, label=NOMBRES[m], color=COL[m])
ax.set_xticks(np.arange(4) + 1.5 * w); ax.set_xticklabels(["Accuracy", "Precision", "Recall", "F1"])
ax.set_ylim(0.85, 1.02); ax.legend(ncol=4, loc="upper center", frameon=False, fontsize=9)
ax.set_title("Comparacion de modelos en el conjunto de prueba")
fig.tight_layout(); fig.savefig(OUT / "comparacion.png", dpi=200); plt.close()

# 3. ROC de todos
fig, ax = plt.subplots(figsize=(4.8, 4))
for m, r in res.items():
    ax.plot(r["roc"]["fpr"], r["roc"]["tpr"], color=COL[m], label=f"{NOMBRES[m]} ({r['roc_auc']:.3f})")
ax.plot([0, 1], [0, 1], "--", color="gray", lw=1)
ax.set_xlabel("Falsos positivos"); ax.set_ylabel("Verdaderos positivos"); ax.legend(frameon=False, fontsize=9)
ax.set_title("Curvas ROC")
fig.tight_layout(); fig.savefig(OUT / "roc.png", dpi=200); plt.close()

# 4. matriz de confusion RF
cm = np.array(res["random_forest"]["matriz_confusion"])
fig, ax = plt.subplots(figsize=(4, 3.5))
ax.imshow(cm, cmap="Blues")
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=16,
                color="white" if cm[i, j] > cm.max() / 2 else "black")
ax.set_xticks([0, 1]); ax.set_xticklabels(["Benigno", "Maligno"]); ax.set_yticks([0, 1]); ax.set_yticklabels(["Benigno", "Maligno"])
ax.set_xlabel("Prediccion"); ax.set_ylabel("Real"); ax.set_title("Matriz de confusion (Random Forest)")
fig.tight_layout(); fig.savefig(OUT / "matriz_rf.png", dpi=200); plt.close()

# 5. importancias RF
imp = res["random_forest"]["importancias"][::-1]
fig, ax = plt.subplots(figsize=(5.5, 3.8))
ax.barh([d["variable"] for d in imp], [d["valor"] for d in imp], color="#1b6ca8")
ax.set_title("Variables mas importantes (Random Forest)")
fig.tight_layout(); fig.savefig(OUT / "importancias.png", dpi=200); plt.close()

for m, r in res.items():
    print(m, {k: r[k] for k in ["accuracy", "precision", "recall", "f1", "roc_auc", "cv_media", "cv_desv"]}, r["matriz_confusion"])
print(res["random_forest"]["importancias"][:5])

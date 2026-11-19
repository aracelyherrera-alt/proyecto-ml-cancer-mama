"""
API del proyecto (backend) hecha con FastAPI.
Endpoints:
  GET  /salud       -> revisa que la API este viva
  GET  /modelos     -> modelos y parametros por defecto
  POST /entrenar    -> entrena con los parametros que se le manden
  GET  /metricas    -> metricas del ultimo entrenamiento
  GET  /variables   -> nombres y estadisticas de las 30 variables
  GET  /ejemplo/{tipo} -> un caso real del dataset (benigno o maligno)
  POST /predecir    -> recibe un caso nuevo y devuelve el diagnostico
  POST /comparar    -> entrena los 4 modelos y los compara (no guarda ninguno)
  GET  /resumen     -> conteo de clases y promedios por clase
"""
from typing import Dict, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ml import entrenamiento as ml

app = FastAPI(title="API Cancer de Mama - Proyecto ML", version="1.0")

# aqui guardamos el modelo que esta en memoria
estado = {"modelo": None, "columnas": None, "metricas": None}


def _cargar_si_existe():
    guardado = ml.cargar_modelo_guardado()
    if guardado:
        estado.update(modelo=guardado["modelo"], columnas=guardado["columnas"],
                      metricas=guardado["metricas"])


_cargar_si_existe()


class PeticionEntrenar(BaseModel):
    modelo: str = "random_forest"
    parametros: Dict = Field(default_factory=dict)
    test_size: float = Field(0.2, ge=0.1, le=0.5)
    semilla: int = 42


class CasoNuevo(BaseModel):
    datos: Dict[str, float]


@app.get("/salud")
def salud():
    return {"estado": "ok", "modelo_entrenado": estado["modelo"] is not None}


@app.get("/modelos")
def modelos():
    return {"disponibles": ml.MODELOS_DISPONIBLES,
            "parametros_por_defecto": ml.PARAMS_POR_DEFECTO}


@app.post("/entrenar")
def entrenar(p: PeticionEntrenar):
    if p.modelo not in ml.MODELOS_DISPONIBLES:
        raise HTTPException(400, f"Modelo no valido. Usa uno de {ml.MODELOS_DISPONIBLES}")
    try:
        metricas = ml.entrenar(p.modelo, p.parametros, p.test_size, p.semilla)
    except Exception as e:  # parametros mal puestos, etc.
        raise HTTPException(400, f"No se pudo entrenar: {e}")
    _cargar_si_existe()
    return metricas


@app.get("/metricas")
def metricas():
    if estado["metricas"] is None:
        raise HTTPException(404, "Todavia no hay un modelo entrenado")
    return estado["metricas"]


@app.get("/variables")
def variables():
    X, _ = ml.cargar_datos()
    desc = X.describe().T
    return [
        {"nombre": c, "min": float(desc.loc[c, "min"]), "max": float(desc.loc[c, "max"]),
         "media": float(desc.loc[c, "mean"])}
        for c in X.columns
    ]


@app.get("/ejemplo/{tipo}")
def ejemplo(tipo: str, indice: Optional[int] = 0):
    """Devuelve una fila real del dataset (tipo = benigno o maligno)."""
    import pandas as pd
    df = pd.read_csv(ml.RUTA_DATOS)
    marca = {"benigno": "B", "maligno": "M"}.get(tipo)
    if marca is None:
        raise HTTPException(400, "tipo debe ser 'benigno' o 'maligno'")
    sub = df[df["diagnosis"] == marca].reset_index(drop=True)
    fila = sub.iloc[(indice or 0) % len(sub)].drop("diagnosis")
    return {k: float(v) for k, v in fila.items()}


@app.post("/predecir")
def predecir(caso: CasoNuevo):
    if estado["modelo"] is None:
        raise HTTPException(409, "Primero hay que entrenar el modelo (POST /entrenar)")
    faltan = [c for c in estado["columnas"] if c not in caso.datos]
    if faltan:
        raise HTTPException(422, f"Faltan variables: {faltan}")
    return ml.predecir(estado["modelo"], estado["columnas"], caso.datos)


class PeticionComparar(BaseModel):
    test_size: float = Field(0.2, ge=0.1, le=0.5)
    semilla: int = 42


@app.post("/comparar")
def comparar(p: PeticionComparar):
    """Entrena los 4 modelos con parametros por defecto y devuelve sus metricas."""
    salida = []
    for nombre in ml.MODELOS_DISPONIBLES:
        r = ml.entrenar(nombre, None, p.test_size, p.semilla, guardar=False)
        r.pop("roc", None)
        salida.append(r)
    return salida


@app.get("/resumen")
def resumen():
    """Cuantos casos hay de cada clase y el promedio de cada variable por clase."""
    import pandas as pd
    df = pd.read_csv(ml.RUTA_DATOS)
    medias = df.groupby("diagnosis").mean(numeric_only=True)
    return {
        "total": int(len(df)),
        "benignos": int((df["diagnosis"] == "B").sum()),
        "malignos": int((df["diagnosis"] == "M").sum()),
        "media_benigno": {k: float(v) for k, v in medias.loc["B"].items()},
        "media_maligno": {k: float(v) for k, v in medias.loc["M"].items()},
    }


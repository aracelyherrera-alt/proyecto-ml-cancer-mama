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
from pathlib import Path
from typing import Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ml import entrenamiento as ml

ESTATICOS = Path(__file__).parent / "static"

DESCRIPCION = """
API del **Predictor de cáncer de mama** (proyecto final de Machine Learning).

Entrena cuatro métodos de clasificación, devuelve sus resultados y estima si un tumor
**parece benigno o maligno** a partir de 30 medidas del núcleo de las células.

> **Importante:** es un ejercicio académico y una herramienta de apoyo para el personal de salud.
> Solo da una estimación: no es un diagnóstico y no reemplaza la valoración de un médico.

Para probar una ruta: ábrela, presiona **Try it out** y luego **Execute**.
"""

ETIQUETAS = [
    {"name": "Estado", "description": "Comprueba que la API está funcionando."},
    {"name": "Entrenamiento", "description": "Elegir, entrenar y comparar los métodos, y ver sus resultados."},
    {"name": "Predicción", "description": "Estimar si un caso nuevo parece benigno o maligno."},
    {"name": "Datos", "description": "Información sobre el dataset y sus 30 medidas."},
]

# docs_url=None: la pagina /docs se define mas abajo, con tema propio y sin depender de internet
app = FastAPI(title="Lazo Rosa · API del predictor de cáncer de mama", version="1.0",
              description=DESCRIPCION, openapi_tags=ETIQUETAS, docs_url=None, redoc_url=None)
app.mount("/estaticos", StaticFiles(directory=ESTATICOS), name="estaticos")


@app.get("/docs", include_in_schema=False)
def documentacion():
    """Documentacion interactiva (Swagger UI) con los archivos guardados en backend/static."""
    pagina = get_swagger_ui_html(
        openapi_url=app.openapi_url, title="Lazo Rosa · Documentación de la API",
        swagger_js_url="/estaticos/swagger/swagger-ui-bundle.js",
        swagger_css_url="/estaticos/swagger/swagger-ui.css",
        swagger_favicon_url="/estaticos/swagger/favicon-32x32.png",
        swagger_ui_parameters={"docExpansion": "list", "defaultModelsExpandDepth": -1,
                               "displayRequestDuration": True},
    )
    html = pagina.body.decode().replace("</head>", '<link rel="stylesheet" href="/estaticos/tema.css"></head>')
    return HTMLResponse(html)

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


@app.get("/salud", tags=["Estado"], summary="Revisar que la API esté activa")
def salud():
    return {"estado": "ok", "modelo_entrenado": estado["modelo"] is not None}


@app.get("/modelos", tags=["Entrenamiento"], summary="Ver los métodos disponibles y sus opciones por defecto")
def modelos():
    return {"disponibles": ml.MODELOS_DISPONIBLES,
            "parametros_por_defecto": ml.PARAMS_POR_DEFECTO}


@app.post("/entrenar", tags=["Entrenamiento"], summary="Entrenar un método")
def entrenar(p: PeticionEntrenar):
    if p.modelo not in ml.MODELOS_DISPONIBLES:
        raise HTTPException(400, f"Modelo no valido. Usa uno de {ml.MODELOS_DISPONIBLES}")
    try:
        metricas = ml.entrenar(p.modelo, p.parametros, p.test_size, p.semilla)
    except Exception as e:  # parametros mal puestos, etc.
        raise HTTPException(400, f"No se pudo entrenar: {e}")
    _cargar_si_existe()
    return metricas


@app.get("/metricas", tags=["Entrenamiento"], summary="Ver los resultados del último entrenamiento")
def metricas():
    if estado["metricas"] is None:
        raise HTTPException(404, "Todavia no hay un modelo entrenado")
    return estado["metricas"]


@app.get("/variables", tags=["Datos"], summary="Ver las 30 medidas con su mínimo, máximo y promedio")
def variables():
    X, _ = ml.cargar_datos()
    desc = X.describe().T
    return [
        {"nombre": c, "min": float(desc.loc[c, "min"]), "max": float(desc.loc[c, "max"]),
         "media": float(desc.loc[c, "mean"])}
        for c in X.columns
    ]


@app.get("/ejemplo/{tipo}", tags=["Datos"], summary="Pedir un caso real de ejemplo (benigno o maligno)")
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


@app.post("/predecir", tags=["Predicción"], summary="Estimar si un caso nuevo parece benigno o maligno")
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


@app.post("/comparar", tags=["Entrenamiento"], summary="Comparar los 4 métodos con los mismos datos")
def comparar(p: PeticionComparar):
    """Entrena los 4 modelos con parametros por defecto y devuelve sus metricas."""
    salida = []
    for nombre in ml.MODELOS_DISPONIBLES:
        r = ml.entrenar(nombre, None, p.test_size, p.semilla, guardar=False)
        r.pop("roc", None)
        salida.append(r)
    return salida


@app.get("/resumen", tags=["Datos"], summary="Ver el resumen del dataset por clase")
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


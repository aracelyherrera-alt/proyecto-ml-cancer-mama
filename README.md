# Predictor de cáncer de mama con Machine Learning

Proyecto final de **Proyectos de Machine Learning** (5to semestre, Desarrollo de Software, período 2026-2).

**Integrantes:** Aracely Estefanía Herrera Regalado · Melany Jessenia Sangoquiza Zapata
**Profesor:** Ing. Sergio Granizo MSc.

## ¿De qué trata?
Una aplicación web que clasifica un tumor de mama como **benigno o maligno** a partir de 30 medidas del núcleo de las células.
Tiene un backend (API con FastAPI) que entrena el modelo y responde predicciones, y un frontend en Python (Streamlit) donde se pueden
configurar los parámetros, ver las métricas y probar un caso nuevo.

## La interfaz
Está hecha con Streamlit (Python). Le pusimos CSS propio (diseño rosa) y los gráficos son SVG generados desde Python, sin librerías de gráficos.
- **Entrenar:** se elige el algoritmo y sus parámetros; muestra accuracy, precisión, recall, F1 y AUC, matriz de confusión, curva ROC e importancia de variables.
- **Comparar:** entrena los 4 algoritmos con los mismos datos y los muestra lado a lado.
- **Probar un caso:** controles para las variables principales, ejemplos reales (benigno / maligno) y resultado con probabilidad, comparado contra los promedios reales.
- **Proyecto:** resumen del dataset, fuente e integrantes.

> Es un proyecto académico. No sirve para diagnosticar a nadie.

## Dataset
Breast Cancer Wisconsin (Diagnostic): 569 casos reales (357 benignos, 212 malignos), 30 variables numéricas.
Fuente: Wolberg, W., Mangasarian, O., Street, N. y Street, W. (1995). *Breast Cancer Wisconsin (Diagnostic)*.
UCI Machine Learning Repository. https://doi.org/10.24432/C5DW2B

## Estructura
```
backend/main.py        API con FastAPI
frontend/              interfaz Streamlit: app.py, graficos.py, estilos.css, static/fuentes/
ml/entrenamiento.py    carga de datos, modelos y métricas
ml/generar_figuras.py  gráficos para el informe
data/wdbc.csv          dataset
modelos/               modelo entrenado (.joblib)
tests/                 pruebas con pytest
```

## Cómo ejecutarlo
Necesitas Python 3.10 o superior.

```bash
# 1. (opcional pero recomendado) entorno virtual
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux / Mac

# 2. instalar librerías
pip install -r requirements.txt

# 3. levantar la API y la interfaz (en dos terminales con el venv activado)
uvicorn backend.main:app --port 8000      # terminal 1 (API)
streamlit run frontend/app.py             # terminal 2 (interfaz)
```
La interfaz se abre en http://localhost:8501. La documentación de la API está en http://localhost:8000/docs.

Pruebas: `pytest -q`

## Endpoints
| Método | Ruta | Descripción |
|---|---|---|
| GET | /salud | Revisa que la API esté activa |
| GET | /modelos | Modelos y parámetros por defecto |
| POST | /entrenar | Entrena con los parámetros enviados |
| GET | /metricas | Métricas del último entrenamiento |
| GET | /variables | Estadísticas de las 30 variables |
| GET | /ejemplo/{tipo} | Un caso real (benigno o maligno) |
| POST | /predecir | Predice un caso nuevo |
| POST | /comparar | Entrena los 4 modelos y los compara |
| GET | /resumen | Conteo de clases y promedios por clase |

Ejemplo:
```bash
curl -X POST http://localhost:8000/entrenar -H "Content-Type: application/json" \
  -d '{"modelo":"svm","parametros":{"C":2.0},"test_size":0.2}'
```

## Resultados (semilla 42, 20% test)
| Modelo | Accuracy | Precisión | Recall | F1 | AUC |
|---|---|---|---|---|---|
| Random Forest | 0.9737 | 1.0000 | 0.9286 | 0.9630 | 0.993 |
| Regresión logística | 0.9649 | 0.9750 | 0.9286 | 0.9512 | 0.996 |
| SVM | 0.9737 | 1.0000 | 0.9286 | 0.9630 | 0.995 |
| KNN | 0.9561 | 0.9744 | 0.9048 | 0.9383 | 0.982 |

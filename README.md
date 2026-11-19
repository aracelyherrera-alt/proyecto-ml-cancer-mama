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
docs/                  se crea al correr ml/generar_figuras.py (figuras del informe)
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

## Limitaciones
- El dataset es pequeño (569 casos) y viene de una sola institución, así que no sabemos qué tan bien funcionaría con pacientes de otros lugares, como Ecuador.
- Las 30 variables ya vienen calculadas por los autores a partir de imágenes. La aplicación **no** analiza imágenes: el usuario escribe o carga las medidas.
- Con solo 114 casos de prueba, uno o dos casos cambian los porcentajes. Por eso los resultados de un solo split no deben tomarse como definitivos (la validación cruzada ayuda a verlo mejor).
- Las clases están algo desbalanceadas (357 benignos y 212 malignos).
- El mejor modelo tuvo 3 falsos negativos de 42 tumores malignos. En un caso real eso sería lo más grave.
- El modelo entrenado se guarda en un solo archivo (`modelos/modelo.joblib`). No hay base de datos ni usuarios, y cada entrenamiento nuevo reemplaza al anterior.
- No usamos GridSearchCV: los parámetros se prueban a mano desde la interfaz.
- **No es un dispositivo médico** y no sirve para diagnosticar. Es solo un ejercicio académico.

## Trabajo futuro
- Ajustar el umbral de decisión para subir el recall del maligno.
- Buscar hiperparámetros con GridSearchCV.
- Probar con otro dataset de otra institución.
- Clasificar las imágenes directamente con una red convolucional (CNN).
- Explicar cada predicción con SHAP.
- Desplegar la aplicación con Docker.

## Reconocimientos
- A los autores del dataset: Wolberg, Street y Mangasarian (Universidad de Wisconsin), y al UCI Machine Learning Repository por publicarlo.
- Al Ing. Sergio Granizo, por la materia Proyectos de Machine Learning.
- A las librerías que usamos: scikit-learn, FastAPI, Streamlit, pandas, NumPy, joblib, requests y pytest.
- A las tipografías Playfair Display y DM Sans (licencia OFL, copias en `frontend/static/fuentes/`).

## Referencias
- Wolberg, W., Mangasarian, O., Street, N. y Street, W. (1995). *Breast Cancer Wisconsin (Diagnostic)*. UCI Machine Learning Repository. https://doi.org/10.24432/C5DW2B
- Street, W. N., Wolberg, W. H. y Mangasarian, O. L. (1993). Nuclear feature extraction for breast tumor diagnosis. *Proceedings of SPIE*, 1905, 861-870.
- Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5-32.
- Cortes, C. y Vapnik, V. (1995). Support-vector networks. *Machine Learning*, 20(3), 273-297.
- Pedregosa, F. et al. (2011). Scikit-learn: Machine learning in Python. *JMLR*, 12, 2825-2830.
- Documentación de FastAPI (https://fastapi.tiangolo.com) y de Streamlit (https://docs.streamlit.io).

## Uso
Proyecto con fines académicos. Los datos pertenecen a sus autores y se usan con la cita indicada arriba.

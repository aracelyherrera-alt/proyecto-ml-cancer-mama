"""
Interfaz gráfica (frontend) hecha en Python con Streamlit.
Solo habla con la API de FastAPI por HTTP (requests), no importa nada del modelo.
Los gráficos se dibujan con SVG desde graficos.py y el diseño está en estilos.css.

Para correrla:  streamlit run frontend/app.py
"""
import os
import sys
from pathlib import Path

import requests
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
import graficos as g  # noqa: E402

API = os.getenv("API_URL", "http://127.0.0.1:8000")
CARPETA = Path(__file__).resolve().parent

st.set_page_config(page_title="Lazo Rosa · Predictor de cáncer de mama", layout="wide")
st.markdown(f"<style>{(CARPETA / 'estilos.css').read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

LAZO = ('<svg viewBox="0 0 24 24"><path fill="currentColor" fill-rule="evenodd" d="M12 1.8c-2.9 0-4.8 2.1-4.8 4.6 0 2.3 1.5 4.1 '
        '3.1 6.4L6.6 20l2.3 1.1 3.1-6.2 3.1 6.2 2.3-1.1-3.7-7.2c1.6-2.3 3.1-4.1 3.1-6.4 0-2.5-1.9-4.6-4.8-4.6Zm0 2.3c1.5 0 2.5 1 2.5 2.4 '
        '0 1.3-.9 2.5-2.5 4.7-1.6-2.2-2.5-3.4-2.5-4.7 0-1.4 1-2.4 2.5-2.4Z"/></svg>')


def html(texto):
    st.markdown(g.limpiar(texto), unsafe_allow_html=True)


def llamar(metodo, ruta, **kw):
    """Llama a la API. Si hay un error lo muestra en pantalla y devuelve None."""
    try:
        r = requests.request(metodo, API + ruta, timeout=180, **kw)
    except requests.exceptions.ConnectionError:
        st.error("No se pudo conectar con la API. Revisa que el backend esté corriendo "
                 "(`uvicorn backend.main:app --port 8000`).")
        st.stop()
    if r.status_code >= 400:
        detalle = r.json().get("detail", r.text) if r.headers.get("content-type", "").startswith("application/json") else r.text
        st.error(detalle)
        return None
    return r.json()


# ------------------------------------------------------------------ datos iniciales
info = llamar("GET", "/modelos")
variables = llamar("GET", "/variables")
resumen = llamar("GET", "/resumen")

if "metricas" not in st.session_state:       # si el backend ya tenía un modelo guardado, lo usamos
    try:
        r = requests.get(API + "/metricas", timeout=30)
        if r.status_code == 200:
            st.session_state["metricas"] = r.json()
    except requests.exceptions.ConnectionError:
        pass
metricas = st.session_state.get("metricas")

# ------------------------------------------------------------------ encabezado (hero)
if metricas:
    tarjeta = (f'<div style="width:120px;margin:0 auto">{g.anillo(metricas["accuracy"], "#dc5f89", 120)}</div>'
               f'<p style="font-family:var(--serif);font-size:1.25rem;font-weight:700;margin:6px 0 2px">{g.NOMBRES[metricas["modelo"]]}</p>'
               f'<p class="vacio-mini">Accuracy en datos de prueba<br>Recall {g.pct(metricas["recall"])} · AUC {metricas["roc_auc"]:.3f}</p>')
else:
    tarjeta = '<p class="vacio-mini">Aún no hay un modelo.<br>Entrena el primero y aparecerá aquí.</p>'
html(f"""
<div class="marca">{LAZO}<span>Lazo Rosa</span></div>
<div class="hero">
  <div>
    <p class="eyebrow">Proyecto final · Machine Learning</p>
    <h1>Predictor de <em>cáncer de mama</em></h1>
    <p class="lead">Un modelo de inteligencia artificial que aprende de {resumen['total']} casos reales y clasifica un tumor como
    <b>benigno</b> o <b>maligno</b>. Entrénalo, compáralo y pruébalo tú misma.</p>
    <div class="hero-datos"><div><b>{resumen['total']}</b><span>casos reales</span></div>
    <div><b>30</b><span>variables</span></div><div><b>4</b><span>algoritmos</span></div></div>
  </div>
  <div class="hero-visual"><div class="orbe"></div>
    <div class="hero-card"><p class="mini">Último modelo entrenado</p>{tarjeta}</div>
    <div class="chip a">Detección temprana</div><div class="chip b">scikit-learn</div></div>
</div>""")

tab_entrenar, tab_comparar, tab_probar, tab_proyecto = st.tabs(
    ["Entrenar", "Comparar modelos", "Probar un caso", "Proyecto"])

# ================================================================== ENTRENAR
with tab_entrenar:
    html('<div class="cab"><p class="eyebrow">Paso 1</p><h2>Entrena tu <em>modelo</em></h2>'
         '<p>Elige un algoritmo, ajusta sus parámetros y mira cómo le va con datos que nunca vio.</p></div>')
    col_conf, col_res = st.columns([1, 2.5], gap="large")

    with col_conf:
        with st.container(border=True):
            html('<div class="sec-titulo">Configuración</div>')
            modelo = st.radio("Algoritmo", info["disponibles"], format_func=lambda x: g.NOMBRES[x])
            params = {}
            if modelo == "random_forest":
                params["n_estimators"] = st.slider("Número de árboles", 10, 500, 100, 10)
                prof = st.slider("Profundidad máxima (0 = sin límite)", 0, 30, 0)
                params["max_depth"] = None if prof == 0 else prof
            elif modelo == "regresion_logistica":
                params["C"] = st.select_slider("C (regularización)", [0.01, 0.1, 1.0, 10.0, 100.0], 1.0)
                params["max_iter"] = st.slider("Iteraciones máximas", 100, 3000, 1000, 100)
            elif modelo == "svm":
                params["C"] = st.select_slider("C (regularización)", [0.01, 0.1, 1.0, 10.0, 100.0], 1.0)
                params["kernel"] = st.selectbox("Kernel", ["rbf", "linear", "poly", "sigmoid"])
            elif modelo == "knn":
                params["n_neighbors"] = st.slider("Número de vecinos (k)", 1, 30, 5)
            test_size = st.slider("Datos de prueba", 0.10, 0.50, 0.20, 0.05, help="El resto se usa para entrenar.")
            semilla = st.number_input("Semilla aleatoria", 0, 9999, 42)
            if st.button("Entrenar modelo", type="primary", use_container_width=True):
                with st.spinner("Entrenando..."):
                    res = llamar("POST", "/entrenar", json={"modelo": modelo, "parametros": params,
                                                            "test_size": test_size, "semilla": int(semilla)})
                if res:
                    st.session_state["metricas"] = res
                    st.rerun()

    with col_res:
        if metricas:
            html(g.resultados(metricas))
        else:
            html(f'<div class="tarjeta vacio">{LAZO}<h3>Todavía no hay resultados</h3>'
                 '<p>Presiona <b>Entrenar modelo</b> y aquí aparecerán las métricas, la matriz de confusión y la curva ROC.</p></div>')

# ================================================================== COMPARAR
with tab_comparar:
    html('<div class="cab"><p class="eyebrow">Paso 2</p><h2>Los cuatro algoritmos, <em>frente a frente</em></h2>'
         '<p>Entrena los cuatro con sus parámetros por defecto y los mismos datos de prueba para ver cuál rinde mejor.</p></div>')
    if st.button("Comparar los 4 modelos", type="primary"):
        with st.spinner("Entrenando los 4 modelos..."):
            st.session_state["comp"] = llamar("POST", "/comparar", json={"test_size": 0.2, "semilla": 42})
    if st.session_state.get("comp"):
        html(g.comparacion(st.session_state["comp"]))

# ================================================================== PROBAR UN CASO
with tab_probar:
    html('<div class="cab"><p class="eyebrow">Paso 3</p><h2>Prueba un <em>caso nuevo</em></h2>'
         '<p>Mueve las medidas del núcleo celular o carga un caso real del dataset, y el modelo entrenado te dirá qué piensa.</p></div>')

    rango = {v["nombre"]: v for v in variables}
    for v in variables:                                   # valores iniciales = promedio
        st.session_state.setdefault(f"v_{v['nombre']}", float(v["media"]))

    def poner(valores):
        for k, val in valores.items():
            st.session_state[f"v_{k}"] = float(val)

    def cargar_ejemplo(tipo):
        import random
        datos = requests.get(f"{API}/ejemplo/{tipo}", params={"indice": random.randint(0, 200)}, timeout=30).json()
        poner(datos)

    col_izq, col_der = st.columns([1.1, 0.9], gap="large")
    with col_izq:
        with st.container(border=True):
            b1, b2, b3 = st.columns(3)
            b1.button("Benigno", on_click=cargar_ejemplo, args=("benigno",), use_container_width=True)
            b2.button("Maligno", on_click=cargar_ejemplo, args=("maligno",), use_container_width=True)
            b3.button("Promedio", on_click=lambda: poner({v["nombre"]: v["media"] for v in variables}),
                      use_container_width=True)
            html('<div class="sec-titulo">Medidas principales</div>')
            c1, c2 = st.columns(2)
            for i, n in enumerate(g.PRINCIPALES):
                with (c1 if i % 2 == 0 else c2):
                    v = rango[n]
                    st.slider(n, float(v["min"]), float(v["max"]), key=f"v_{n}", step=float(v["max"] - v["min"]) / 300,
                              format="%.4g")
            with st.expander("Ver las otras 24 variables"):
                resto = [v["nombre"] for v in variables if v["nombre"] not in g.PRINCIPALES]
                e1, e2 = st.columns(2)
                for i, n in enumerate(resto):
                    with (e1 if i % 2 == 0 else e2):
                        st.number_input(n, key=f"v_{n}", format="%.5f")
            if st.button("Predecir", type="primary", use_container_width=True):
                valores = {v["nombre"]: float(st.session_state[f"v_{v['nombre']}"]) for v in variables}
                res = llamar("POST", "/predecir", json={"datos": valores})
                if res:
                    st.session_state["pred"] = (res, valores)

    with col_der:
        if st.session_state.get("pred"):
            res, valores = st.session_state["pred"]
            html(g.resultado_caso(res, valores, resumen))
        else:
            html(f'<div class="tarjeta vacio">{LAZO}<h3>Resultado</h3>'
                 '<p>Carga un ejemplo o mueve los controles y presiona <b>Predecir</b>.</p></div>')

# ================================================================== PROYECTO
with tab_proyecto:
    html('<div class="cab"><p class="eyebrow">Sobre el trabajo</p><h2>Datos y <em>proyecto</em></h2></div>')
    c1, c2 = st.columns([1.4, 1], gap="large")
    with c1:
        html(f"""<div class="tarjeta"><h3>El dataset</h3>
        <div style="display:grid;grid-template-columns:210px 1fr;gap:24px;align-items:center">
        {g.dona(resumen)}
        <div><p><b>Breast Cancer Wisconsin (Diagnostic)</b>. Las 30 variables se calcularon a partir de imágenes digitalizadas de
        punciones con aguja fina (FNA) de masas mamarias. Para 10 medidas del núcleo celular (radio, textura, perímetro, área,
        suavidad, compacidad, concavidad, puntos cóncavos, simetría y dimensión fractal) se guarda el promedio, el error estándar
        y el peor valor.</p>
        <p class="fuente">Wolberg, W., Mangasarian, O., Street, N. y Street, W. (1995). UCI Machine Learning Repository.
        <a href="https://doi.org/10.24432/C5DW2B" target="_blank">doi.org/10.24432/C5DW2B</a></p></div></div></div>""")
    with c2:
        html("""<div class="tarjeta"><h3>Cómo funciona</h3><ol class="pasos">
        <li><span><b>Esta interfaz</b> (Streamlit, en Python) envía los parámetros a la API.</span></li>
        <li><span><b>La API (FastAPI)</b> entrena el modelo con scikit-learn y lo guarda.</span></li>
        <li><span><b>Con el modelo guardado</b> se predicen casos nuevos y se devuelven las probabilidades.</span></li></ol>
        <p class="aviso">Este proyecto es un ejercicio académico. Sus resultados no reemplazan un diagnóstico médico.</p></div>""")
    html("""<div class="tarjeta"><h3>Integrantes</h3><div class="personas">
    <div class="persona"><span class="avatar">AH</span><div><b>Aracely Estefanía Herrera Regalado</b><small>Desarrollo de Software · 5to semestre</small></div></div>
    <div class="persona"><span class="avatar lila">MS</span><div><b>Melany Jessenia Sangoquiza Zapata</b><small>Desarrollo de Software · 5to semestre</small></div></div></div>
    <p class="materia">Proyectos de Machine Learning · Ing. Sergio Granizo MSc.</p></div>""")

html('<div class="pie">Proyecto final de Proyectos de Machine Learning · 2026<br>'
     '<small>Solo con fines académicos. No sustituye la valoración de un profesional de la salud.</small></div>')

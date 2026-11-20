"""Interfaz del proyecto (Streamlit). Habla con la API de FastAPI.
Para correrla: streamlit run frontend/app.py"""
import os
import sys
from pathlib import Path

import requests
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
import graficos as g

API = os.getenv("API_URL", "http://127.0.0.1:8000")
CARPETA = Path(__file__).resolve().parent

st.set_page_config(page_title="Lazo Rosa · Predictor de cáncer de mama", layout="wide", initial_sidebar_state="expanded")
st.markdown(f"<style>{(CARPETA / 'estilos.css').read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

LAZO = ('<svg viewBox="0 0 24 24"><path fill="currentColor" fill-rule="evenodd" d="M12 1.8c-2.9 0-4.8 2.1-4.8 4.6 0 2.3 1.5 4.1 '
        '3.1 6.4L6.6 20l2.3 1.1 3.1-6.2 3.1 6.2 2.3-1.1-3.7-7.2c1.6-2.3 3.1-4.1 3.1-6.4 0-2.5-1.9-4.6-4.8-4.6Zm0 2.3c1.5 0 2.5 1 2.5 2.4 '
        '0 1.3-.9 2.5-2.5 4.7-1.6-2.2-2.5-3.4-2.5-4.7 0-1.4 1-2.4 2.5-2.4Z"/></svg>')

LAZO_GRANDE = LAZO.replace('<svg viewBox="0 0 24 24">', '<svg class="lazo-svg" viewBox="0 0 24 24">')


def html(texto):
    st.markdown(g.limpiar(texto), unsafe_allow_html=True)


def llamar(metodo, ruta, **kw):
    # llama a la API; si falla muestra el error
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


AYUDA_C = ("Valores bajos hacen un modelo más simple y flexible; valores altos hacen que se ajuste "
           "más a los casos con los que aprende.")


# datos que vienen de la API
info = llamar("GET", "/modelos")
variables = llamar("GET", "/variables")
resumen = llamar("GET", "/resumen")

metricas = st.session_state.get("metricas")

# menú lateral
PAGINAS = ["Inicio", "Entrenar modelo", "Comparar modelos", "Probar un caso", "Información"]


def ir(pagina):
    st.session_state["pagina"] = pagina


def elegir_modelo(k):
    st.session_state["modelo"] = k


st.session_state.setdefault("pagina", "Inicio")
with st.sidebar:
    html(f'''<div class="lado-marca">{LAZO_GRANDE}<div class="lado-nombre">Lazo Rosa</div>
    <div class="lado-lema">Cuídate, siempre</div></div>''')
    for i, nombre in enumerate(PAGINAS):
        st.button(nombre, key=f"nav_{i}", use_container_width=True, on_click=ir, args=(nombre,),
                  type="primary" if st.session_state["pagina"] == nombre else "secondary")
    html('<div class="lado-frase">La detección temprana<br>salva vidas</div>')
pagina = st.session_state["pagina"]

# --- Inicio
if pagina == "Inicio":
    comp = st.session_state.get("comp")
    html(f'''
    <div class="hero">
      <div>
        <p class="eyebrow">Proyecto final · Machine Learning</p>
        <h1 class="h1-fino">Predictor inteligente de</h1>
        <h1 class="h1-grande">Cáncer de Mama</h1>
        <p class="lema-hero">Tecnología + Salud = Bienestar</p>
        <p class="lead">Una inteligencia artificial que aprendió de {resumen['total']} casos reales de tumores de mama y estima si un tumor
        parece <b>benigno</b> o <b>maligno</b>. Entrénala, compárala y pruébala.</p>
        <div class="hero-datos"><div><b>{resumen['total']}</b><span>casos reales</span></div>
        <div><b>30</b><span>medidas</span></div><div><b>4</b><span>métodos</span></div></div>
      </div>
      <div class="hero-visual"><div class="orbe"></div><div class="lazo-hero">{LAZO_GRANDE}</div>
        <div class="chip a">Detección temprana</div><div class="chip b">scikit-learn</div></div>
    </div>''')

    b1, b2, _ = st.columns([1, 1, 2.2], gap="medium")
    b1.button("Probar un caso", key="ir_probar", type="primary", use_container_width=True, on_click=ir, args=("Probar un caso",))
    b2.button("Entrenar un modelo", key="ir_entrenar", use_container_width=True, on_click=ir, args=("Entrenar modelo",))

    def paso(hecho, n, titulo, detalle):
        return (f'<li class="{"hecho" if hecho else "pend"}"><span class="marca-p">{"" if hecho else n}</span>'
                f'<div><b>{titulo}</b><small>{detalle}</small></div></li>')

    if metricas:
        d1 = f'{g.NOMBRES[metricas["modelo"]]} · {g.pct(metricas["accuracy"])} de aciertos'
    else:
        d1 = "Elige un método y mira cuántos casos acierta."
    if comp:
        mejor = max(comp, key=lambda m: m["accuracy"])
        d2 = f'Mejor por ahora: {g.NOMBRES[mejor["modelo"]]} ({g.pct(mejor["accuracy"])})'
    else:
        d2 = "Pon los 4 métodos frente a frente."
    d3 = "Ya probaste un caso. Puedes cargar otro." if st.session_state.get("pred") else "Mueve las medidas y mira la estimación."
    progreso = (paso(bool(metricas), 1, "Entrenar un modelo", d1) + paso(bool(comp), 2, "Comparar los 4 modelos", d2)
                + paso(bool(st.session_state.get("pred")), 3, "Probar un caso nuevo", d3))
    metodos = "".join(f'<li><i style="background:{g.COLORES[k]}"></i><div><b>{g.NOMBRES_SENCILLOS[k]}</b>'
                      f'<small>{g.DESCRIPCION_MODELOS[k]}</small></div></li>' for k in g.NOMBRES)
    html(f'''<div class="fila3">
      <div class="tarjeta"><div class="tit-card"><h3>Tu avance</h3></div><ul class="progreso">{progreso}</ul></div>
      <div class="tarjeta"><div class="tit-card"><h3>Los 4 métodos</h3></div><ul class="metodos">{metodos}</ul></div>
      <div class="tarjeta mision"><div class="tit-card"><h3>Nuestra misión</h3></div>
        <p>Mostrar, de forma sencilla, cómo el Machine Learning puede ayudar a distinguir un tumor benigno de uno maligno,
        y recordar lo más importante: <b>tu salud</b>.</p>
        <p class="nota-mini">Es un ejercicio académico: no reemplaza la valoración de un médico.</p>
        <div class="lazo-mision">{LAZO_GRANDE}</div></div>
    </div>''')

# --- Entrenar modelo
if pagina == "Entrenar modelo":
    html('<div class="cab"><p class="eyebrow">Paso 1</p><h2>Entrena tu <em>modelo</em></h2>'
         '<p>Elige un método, ajusta sus opciones y mira cómo le va con casos que nunca vio.</p></div>')

    with st.container(border=True):
        html('<div class="sec-titulo"><span class="num">1</span>Elige el método</div>')
        modelo = st.session_state.get("modelo")          # al inicio no hay ninguno elegido
        cols_m = st.columns(len(info["disponibles"]), gap="medium")
        for col, k in zip(cols_m, info["disponibles"]):
            col.button(g.NOMBRES_SENCILLOS[k], key=f"m_{k}", use_container_width=True, on_click=elegir_modelo, args=(k,),
                       type="primary" if modelo == k else "secondary")
        if modelo is None:
            html('<p class="desc-modelo">Elige uno de los cuatro métodos para ver sus opciones y poder entrenarlo.</p>')
        else:
            html(f'<p class="desc-modelo"><b>{g.NOMBRES[modelo]}.</b> {g.DESCRIPCION_MODELOS[modelo]}</p>')
            params = {}
            with st.expander("Opciones avanzadas (no hace falta tocarlas)"):
                st.caption("Ya vienen con valores recomendados. Cámbialas solo si conoces Machine Learning.")
                ca, cb = st.columns(2, gap="large")
                if modelo == "random_forest":
                    with ca:
                        params["n_estimators"] = st.slider("Cantidad de árboles de decisión", 10, 500, 100, 10,
                                                           help="Más árboles suelen dar resultados más estables, pero tardan más.")
                    with cb:
                        prof = st.slider("Detalle máximo de cada árbol (0 = sin límite)", 0, 30, 0,
                                         help="Qué tan detallado puede ser cada árbol de decisión.")
                        params["max_depth"] = None if prof == 0 else prof
                elif modelo == "regresion_logistica":
                    with ca:
                        params["C"] = st.select_slider("Rigidez del modelo (C)", [0.01, 0.1, 1.0, 10.0, 100.0], 1.0, help=AYUDA_C)
                    with cb:
                        params["max_iter"] = st.slider("Intentos máximos de ajuste", 100, 3000, 1000, 100,
                                                       help="Cuántas veces puede intentar mejorar antes de detenerse.")
                elif modelo == "svm":
                    with ca:
                        params["C"] = st.select_slider("Rigidez del modelo (C)", [0.01, 0.1, 1.0, 10.0, 100.0], 1.0, help=AYUDA_C)
                    with cb:
                        params["kernel"] = st.selectbox("Forma de separar los casos (kernel)", ["rbf", "linear", "poly", "sigmoid"],
                                                        help="rbf: frontera curva. linear: línea recta. poly y sigmoid: otras formas curvas.")
                elif modelo == "knn":
                    with ca:
                        params["n_neighbors"] = st.slider("Cuántos casos parecidos se consultan (k)", 1, 30, 5,
                                                          help="Cuántos casos parecidos se consultan para decidir.")
                cc, cd = st.columns(2, gap="large")
                with cc:
                    test_size = st.slider("Casos reservados para la prueba", 0.10, 0.50, 0.20, 0.05,
                                          help="Parte de los casos que el programa no ve al aprender y que se usan para evaluarlo. El resto se usa para entrenar.")
                with cd:
                    semilla = st.number_input("Número del sorteo (semilla)", 0, 9999, 42,
                                              help="Fija cómo se sortean los casos. Con el mismo número se obtienen los mismos resultados.")
            bt, _ = st.columns([1, 2])
            with bt:
                entrenar = st.button("Entrenar este método", type="primary", use_container_width=True)
            if entrenar:
                with st.spinner("Entrenando..."):
                    res = llamar("POST", "/entrenar", json={"modelo": modelo, "parametros": params,
                                                            "test_size": test_size, "semilla": int(semilla)})
                if res:
                    st.session_state["metricas"] = res
                    st.rerun()

    if metricas:
        html(g.resultados(metricas))
        html(g.glosario())
    elif st.session_state.get("modelo") is None:
        html(f'<div class="tarjeta vacio">{LAZO}<h3>Primero elige un método</h3>'
             '<p>Cuando elijas uno, podrás entrenarlo y aquí aparecerán los resultados.</p></div>')
    else:
        html(f'<div class="tarjeta vacio">{LAZO}<h3>Todavía no hay resultados</h3>'
             '<p>Presiona <b>Entrenar este método</b> y aquí aparecerán los resultados, los aciertos y errores del programa y la curva ROC.</p></div>')

# --- Comparar modelos
if pagina == "Comparar modelos":
    html('<div class="cab"><p class="eyebrow">Paso 2</p><h2>Los cuatro algoritmos, <em>frente a frente</em></h2>'
         '<p>Entrena los cuatro métodos con sus opciones por defecto y los mismos casos de prueba para ver cuál rinde mejor.</p></div>')
    if st.button("Comparar los 4 modelos", type="primary"):
        with st.spinner("Entrenando los 4 modelos..."):
            st.session_state["comp"] = llamar("POST", "/comparar", json={"test_size": 0.2, "semilla": 42})
    if st.session_state.get("comp"):
        html(g.comparacion(st.session_state["comp"]))
        html('<p class="nota-eje">¿Dudas con algún nombre? Mira «¿Qué significan los números?» al final de <b>Entrenar modelo</b>.</p>')

# --- Probar un caso
if pagina == "Probar un caso":
    html('<div class="cab"><p class="eyebrow">Paso 3</p><h2>Prueba un <em>caso nuevo</em></h2>'
         '<p>Estas medidas salen del análisis de una muestra en laboratorio, no se pueden medir en casa. Mueve los controles o carga un ejemplo real y el programa estimará si el tumor parece benigno o maligno.</p></div>')

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

    with st.container(border=True):
        html('<div class="sec-titulo"><span class="num">1</span>Carga un ejemplo o mueve las medidas</div>')
        b1, b2, b3 = st.columns(3, gap="medium")
        b1.button("Ejemplo benigno", on_click=cargar_ejemplo, args=("benigno",), use_container_width=True)
        b2.button("Ejemplo maligno", on_click=cargar_ejemplo, args=("maligno",), use_container_width=True)
        b3.button("Valores promedio", on_click=lambda: poner({v["nombre"]: v["media"] for v in variables}),
                  use_container_width=True)
        html('<div class="sec-titulo sub">Medidas principales de la célula</div>')
        cols = st.columns(3, gap="large")
        for i, n in enumerate(g.PRINCIPALES):
            with cols[i % 3]:
                v = rango[n]
                st.slider(g.etiqueta(n), float(v["min"]), float(v["max"]), key=f"v_{n}", step=float(v["max"] - v["min"]) / 300,
                          format="%.4g", help=g.ayuda(n))
        with st.expander("Ver las otras 24 medidas"):
            resto = [v["nombre"] for v in variables if v["nombre"] not in g.PRINCIPALES]
            ecols = st.columns(3, gap="large")
            for i, n in enumerate(resto):
                with ecols[i % 3]:
                    st.number_input(g.etiqueta(n), key=f"v_{n}", format="%.5f", help=g.ayuda(n))
        bp, _ = st.columns([1, 2])
        with bp:
            predecir = st.button("Predecir", type="primary", use_container_width=True)
        if predecir:
            valores = {v["nombre"]: float(st.session_state[f"v_{v['nombre']}"]) for v in variables}
            res = llamar("POST", "/predecir", json={"datos": valores})
            if res:
                st.session_state["pred"] = (res, valores)

    if st.session_state.get("pred"):
        res, valores = st.session_state["pred"]
        html(g.resultado_caso(res, valores, resumen))
    else:
        html(f'<div class="tarjeta vacio">{LAZO}<h3>Resultado</h3>'
             '<p>Carga un ejemplo o mueve los controles y presiona <b>Predecir</b>.</p></div>')

# --- Información
if pagina == "Información":
    html('<div class="cab"><p class="eyebrow">Sobre el trabajo</p><h2>Datos y <em>proyecto</em></h2></div>')
    html(f"""<div class="tarjeta"><h3>El dataset</h3>
    <div class="ds-grid">{g.dona(resumen)}
    <div><p><b>Breast Cancer Wisconsin (Diagnostic)</b>. Las 30 medidas se calcularon a partir de imágenes de
    muestras tomadas con una aguja fina (punción) de bultos en la mama. De cada célula se miden 10 características (radio, textura, perímetro, tamaño,
    suavidad, compacidad, hundimientos del borde, puntos de hundimiento, simetría e irregularidad del contorno) y de cada una se guarda el promedio,
    la variación y el valor más alto.</p>
    <p class="fuente">Wolberg, W., Mangasarian, O., Street, N. y Street, W. (1995). UCI Machine Learning Repository.
    <a href="https://doi.org/10.24432/C5DW2B" target="_blank">doi.org/10.24432/C5DW2B</a></p></div></div></div>""")
    html("""<div class="tarjeta"><h3>Cómo funciona</h3><ol class="pasos pasos-fila">
      <li><span><b>Esta interfaz</b> (Streamlit, en Python) envía los parámetros a la API.</span></li>
      <li><span><b>La API (FastAPI)</b> entrena el modelo con scikit-learn y lo guarda.</span></li>
      <li><span><b>Con el modelo guardado</b> se predicen casos nuevos y se devuelven las probabilidades.</span></li></ol>
      <p class="aviso">Este proyecto es un ejercicio académico. Sus resultados son solo una estimación y no reemplazan la valoración de un médico.</p></div>""")
    html("""<div class="tarjeta"><h3>Integrantes</h3><div class="personas">
    <div class="persona"><span class="avatar">AH</span><div><b>Aracely Estefanía Herrera Regalado</b><small>Desarrollo de Software · 5to semestre</small></div></div>
    <div class="persona"><span class="avatar lila">MS</span><div><b>Melany Jessenia Sangoquiza Zapata</b><small>Desarrollo de Software · 5to semestre</small></div></div></div>
    <p class="materia">Proyectos de Machine Learning · Ing. Sergio Granizo MSc.</p></div>""")

html('<div class="pie">Proyecto final de Proyectos de Machine Learning · 2026<br>'
     '<small>Solo con fines académicos. No sustituye la valoración de un profesional de la salud.</small></div>')

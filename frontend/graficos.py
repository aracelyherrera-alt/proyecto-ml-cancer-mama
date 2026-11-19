"""
Gráficos de la interfaz, dibujados con SVG/HTML desde Python (sin librerías de gráficos).
Cada función devuelve un texto HTML que app.py muestra con st.markdown(..., unsafe_allow_html=True).
Las animaciones las hace el CSS (estilos.css), no hay JavaScript.
"""
import math
import re

NOMBRES = {"random_forest": "Random Forest", "regresion_logistica": "Regresión logística",
           "svm": "SVM", "knn": "KNN"}
COLORES = {"random_forest": "#ea7fa2", "regresion_logistica": "#8f74d6", "svm": "#4fae91", "knn": "#f2b57f"}
PRINCIPALES = ["worst_area", "worst_concave_points", "worst_radius",
               "mean_concave_points", "worst_perimeter", "mean_radius"]


def limpiar(html: str) -> str:
    """Quita saltos de línea y sangrías: Markdown trataría líneas con 4 espacios como código."""
    return re.sub(r"\s*\n\s*", " ", html).strip()


def pct(v, d=1):
    return f"{v * 100:.{d}f}%"


def pct_corto(v):
    return re.sub(r"\.0%$", "%", f"{v * 100:.1f}%")


def fmt_num(v):
    a = abs(v)
    return f"{v:.0f}" if a >= 100 else f"{v:.1f}" if a >= 10 else f"{v:.2f}" if a >= 1 else f"{v:.4f}"


# ------------------------------------------------------------------ anillos de métricas
def anillo(valor, color, tam=96):
    c = 2 * math.pi * 40
    return (f'<svg viewBox="0 0 100 100" width="{tam}" height="{tam}">'
            f'<circle cx="50" cy="50" r="40" fill="none" stroke="#ffe3ec" stroke-width="9"/>'
            f'<circle class="anim-anillo" cx="50" cy="50" r="40" fill="none" stroke="{color}" stroke-width="9" '
            f'stroke-linecap="round" stroke-dasharray="{c:.1f}" stroke-dashoffset="{c * (1 - valor):.1f}" '
            f'style="--full:{c:.1f}" transform="rotate(-90 50 50)"/>'
            f'<text x="50" y="56" text-anchor="middle" font-family="Playfair Display,serif" font-weight="700" '
            f'font-size="19" fill="#3f2136">{pct_corto(valor)}</text></svg>')


def tarjetas_kpi(m):
    kpis = [("Accuracy", m["accuracy"], "aciertos totales", "#dc5f89"),
            ("Precisión", m["precision"], "de los que dijo maligno, ¿cuántos lo eran?", "#8f74d6"),
            ("Recall", m["recall"], "de los malignos, ¿cuántos detectó?", "#4fae91"),
            ("F1", m["f1"], "equilibrio entre precisión y recall", "#f2a366"),
            ("AUC-ROC", m["roc_auc"], f"validación cruzada: {pct(m['cv_media'])} ± {pct(m['cv_desv'])}", "#c24572")]
    return '<div class="kpis">' + "".join(
        f'<div class="kpi">{anillo(v, col)}<div class="t">{t}</div><div class="d">{d}</div></div>'
        for t, v, d, col in kpis) + "</div>"


# ------------------------------------------------------------------ matriz de confusión
def matriz_confusion(cm):
    mx = max(max(f) for f in cm) or 1

    def celda(v, txt):
        a = 0.1 + 0.9 * v / mx
        col = "#fff" if a > 0.5 else "#3f2136"
        return (f'<div class="celda" style="background:rgba(220,95,137,{a:.2f});color:{col}">{v}<small>{txt}</small></div>')

    fn = cm[1][0]
    if fn == 0:
        nota = '<div class="nota ok">No se le pasó ningún tumor maligno (0 falsos negativos).</div>'
    else:
        nota = (f'<div class="nota warn">Se le pasaron <b>{fn}</b> tumor(es) maligno(s) (falsos negativos). '
                'Es el error más delicado en un problema médico.</div>')
    return ('<div class="matriz"><span></span><span class="eje">Predicho benigno</span><span class="eje">Predicho maligno</span>'
            f'<span class="eje v">Real benigno</span>{celda(cm[0][0], "acertó")}{celda(cm[0][1], "falsa alarma")}'
            f'<span class="eje v">Real maligno</span>{celda(cm[1][0], "se le pasó")}{celda(cm[1][1], "acertó")}</div>{nota}')


# ------------------------------------------------------------------ curva ROC
def curva_roc(roc, auc):
    x0, x1, y0, y1 = 50, 400, 274, 14
    px = lambda x: x0 + x * (x1 - x0)
    py = lambda y: y0 - y * (y0 - y1)
    pts = [f"{px(x):.1f},{py(y):.1f}" for x, y in zip(roc["fpr"], roc["tpr"])]
    linea = "M" + " L".join(pts)
    area = f"{linea} L{px(1)},{py(0)} L{px(0)},{py(0)} Z"
    rejilla = ""
    for t in (0, .25, .5, .75, 1):
        rejilla += (f'<line x1="{x0}" x2="{x1}" y1="{py(t)}" y2="{py(t)}" stroke="#ffe3ec"/>'
                    f'<text x="{x0 - 10}" y="{py(t) + 4}" text-anchor="end" font-size="11" fill="#6b4560">{t}</text>'
                    f'<text x="{px(t)}" y="{y0 + 18}" text-anchor="middle" font-size="11" fill="#6b4560">{t}</text>')
    return (f'<svg viewBox="0 0 420 320" width="100%"><defs><linearGradient id="gr" x1="0" y1="0" x2="0" y2="1">'
            '<stop offset="0" stop-color="#ea7fa2" stop-opacity=".45"/><stop offset="1" stop-color="#c7b3f0" stop-opacity=".08"/>'
            f'</linearGradient></defs>{rejilla}<path d="{area}" fill="url(#gr)"/>'
            f'<line x1="{px(0)}" y1="{py(0)}" x2="{px(1)}" y2="{py(1)}" stroke="#b59aa9" stroke-dasharray="5 5"/>'
            f'<path d="{linea}" fill="none" stroke="#c24572" stroke-width="3.2" stroke-linejoin="round" stroke-linecap="round"/>'
            f'<text x="{px(.52)}" y="{py(.22)}" font-family="Playfair Display,serif" font-weight="700" font-size="22" fill="#c24572">AUC {auc:.3f}</text>'
            f'<text x="{(x0 + x1) / 2}" y="312" text-anchor="middle" font-size="11.5" fill="#6b4560">Tasa de falsos positivos</text>'
            f'<text transform="translate(12 {(y0 + y1) / 2}) rotate(-90)" text-anchor="middle" font-size="11.5" fill="#6b4560">Tasa de verdaderos positivos</text></svg>')


# ------------------------------------------------------------------ importancia de variables
def barras_importancia(imp):
    mx = max(i["valor"] for i in imp) or 1e-9
    filas = "".join(
        f'<div class="bh"><span class="nombre">{i["variable"]}</span><div class="pista">'
        f'<div class="relleno" style="width:{i["valor"] / mx * 100:.1f}%"></div></div><span class="val">{i["valor"]:.3f}</span></div>'
        for i in imp)
    return f'<div class="barras-h">{filas}</div>'


# ------------------------------------------------------------------ bloque completo de resultados
def resultados(m):
    params = " ".join(f"<code>{k}={'None' if v is None else v}</code>" for k, v in m["parametros"].items())
    imp = (f'<div class="tarjeta"><h4>Variables que más pesan en el modelo (top 10)</h4>{barras_importancia(m["importancias"])}</div>'
           if m.get("importancias") else "")
    return limpiar(f"""
    <div>
      <div class="tarjeta">
        <div class="res-cab"><h3>{NOMBRES[m['modelo']]}</h3><p>{m['n_train']} casos de entrenamiento · {m['n_test']} de prueba</p></div>
        <p class="params">Parámetros: {params}</p>
        {tarjetas_kpi(m)}
      </div>
      <div class="dos">
        <div class="tarjeta"><h4>Matriz de confusión</h4>{matriz_confusion(m['matriz_confusion'])}</div>
        <div class="tarjeta"><h4>Curva ROC</h4>{curva_roc(m['roc'], m['roc_auc'])}</div>
      </div>
      {imp}
    </div>""")


# ------------------------------------------------------------------ comparación de modelos
def comparacion(lista):
    metricas = [("accuracy", "Accuracy"), ("precision", "Precisión"), ("recall", "Recall"), ("f1", "F1")]
    x0, x1, y0, y1, lo, hi = 64, 840, 330, 40, 0.85, 1.0
    py = lambda v: y0 - (min(max(v, lo), hi) - lo) / (hi - lo) * (y0 - y1)
    ancho_g = (x1 - x0) / len(metricas)
    bw, gap = 30, 8
    total = len(lista) * bw + (len(lista) - 1) * gap
    s = (f'<svg viewBox="0 0 860 372" width="100%"><defs><clipPath id="cp"><rect x="{x0}" y="{y1 - 12}" '
         f'width="{x1 - x0}" height="{y0 - y1 + 12}"/></clipPath></defs>')
    for t in (0.85, 0.9, 0.95, 1.0):
        s += (f'<line x1="{x0}" x2="{x1}" y1="{py(t)}" y2="{py(t)}" stroke="#ffe3ec"/>'
              f'<text x="{x0 - 10}" y="{py(t) + 4}" text-anchor="end" font-size="12" fill="#6b4560">{t:.2f}</text>')
    for gi, (k, nombre) in enumerate(metricas):
        xg = x0 + gi * ancho_g + (ancho_g - total) / 2
        for i, m in enumerate(lista):
            x, y = xg + i * (bw + gap), py(m[k])
            s += (f'<g clip-path="url(#cp)"><rect class="barra-svg" x="{x}" y="{y}" width="{bw}" height="{y0 - y + 8}" rx="8" '
                  f'fill="{COLORES[m["modelo"]]}" style="animation-delay:{(gi * 4 + i) * 40}ms">'
                  f'<title>{NOMBRES[m["modelo"]]} · {nombre}: {m[k]:.4f}</title></rect></g>'
                  f'<text x="{x + bw / 2}" y="{y - 7}" text-anchor="middle" font-size="10.5" font-weight="700" fill="#3f2136">{m[k]:.3f}</text>')
        s += (f'<text x="{x0 + gi * ancho_g + ancho_g / 2}" y="{y0 + 30}" text-anchor="middle" '
              f'font-family="Playfair Display,serif" font-weight="700" font-size="17" fill="#3f2136">{nombre}</text>')
    s += '</svg><p class="nota-eje">El eje vertical empieza en 0.85 para que se noten las diferencias.</p>'

    cols = [("accuracy", "Accuracy"), ("precision", "Precisión"), ("recall", "Recall"), ("f1", "F1"),
            ("roc_auc", "AUC"), ("cv_media", "CV 5 folds")]
    mejor = {k: max(m[k] for m in lista) for k, _ in cols}
    cab = "<th>Modelo</th>" + "".join(f"<th>{n}</th>" for _, n in cols)
    cuerpo = ""
    for m in lista:
        cuerpo += (f'<tr><td><i style="display:inline-block;width:10px;height:10px;border-radius:3px;'
                   f'background:{COLORES[m["modelo"]]};margin-right:8px"></i>{NOMBRES[m["modelo"]]}</td>')
        cuerpo += "".join(f'<td class="{"mejor" if abs(m[k] - mejor[k]) < 1e-9 else ""}">{m[k]:.4f}</td>' for k, _ in cols) + "</tr>"
    tabla = f'<div class="tabla-wrap"><table class="tabla"><thead><tr>{cab}</tr></thead><tbody>{cuerpo}</tbody></table></div>'

    top = [NOMBRES[m["modelo"]] for m in lista if abs(m["accuracy"] - mejor["accuracy"]) < 1e-9]
    frase = (f'<b>{" y ".join(top)}</b> empatan con el mayor accuracy ({pct(mejor["accuracy"], 2)})' if len(top) > 1
             else f'<b>{top[0]}</b> tiene el mayor accuracy ({pct(mejor["accuracy"], 2)})')
    leyenda = "".join(f'<span><i style="background:{COLORES[m["modelo"]]}"></i>{NOMBRES[m["modelo"]]}</span>' for m in lista)
    return limpiar(f"""
    <div>
      <div class="ganador"><span class="copa">🏆</span><p>{frase}<br><small>Ojo: con pocos datos de prueba las diferencias son de uno o dos casos, mira también la validación cruzada.</small></p></div>
      <div class="tarjeta"><div class="leyenda">{leyenda}</div>{s}</div>
      <div class="tarjeta">{tabla}</div>
    </div>""")


# ------------------------------------------------------------------ resultado de un caso nuevo
def medidor(pm, es_maligno):
    r = 110
    largo = math.pi * r
    ang = math.pi * (1 - pm)
    mx, my = 150 + r * math.cos(ang), 150 - r * math.sin(ang)
    borde = "#cf3f68" if es_maligno else "#4fae91"
    return (f'<svg class="gauge" viewBox="0 0 300 184"><defs><linearGradient id="gg" x1="0" x2="1">'
            '<stop offset="0" stop-color="#4fae91"/><stop offset=".5" stop-color="#f2b57f"/><stop offset="1" stop-color="#cf3f68"/></linearGradient></defs>'
            '<path d="M40 150 A110 110 0 0 1 260 150" fill="none" stroke="#ffe3ec" stroke-width="20" stroke-linecap="round"/>'
            f'<path class="anim-anillo" d="M40 150 A110 110 0 0 1 260 150" fill="none" stroke="url(#gg)" stroke-width="20" stroke-linecap="round" '
            f'stroke-dasharray="{largo:.1f}" stroke-dashoffset="{largo * (1 - pm):.1f}" style="--full:{largo:.1f}"/>'
            f'<circle cx="{mx:.1f}" cy="{my:.1f}" r="11" fill="#fff" stroke="{borde}" stroke-width="5"/>'
            f'<text x="150" y="140" text-anchor="middle" font-family="Playfair Display,serif" font-weight="700" font-size="40" fill="#3f2136">{pct(pm)}</text>'
            '<text x="150" y="162" text-anchor="middle" font-size="12" fill="#6b4560">probabilidad de maligno</text>'
            '<text x="40" y="180" text-anchor="middle" font-size="10.5" fill="#6b4560">0%</text>'
            '<text x="260" y="180" text-anchor="middle" font-size="10.5" fill="#6b4560">100%</text></svg>')


def resultado_caso(res, valores, resumen):
    es_m = res["diagnostico"] == "Maligno"
    conf = res["probabilidad_maligno"] if es_m else res["probabilidad_benigno"]
    lo, hi = -0.6, 2.0
    pos = lambda t: min(97, max(3, (t - lo) / (hi - lo) * 100))
    cerca, filas = 0, ""
    for n in PRINCIPALES:
        b, mm = resumen["media_benigno"][n], resumen["media_maligno"][n]
        t = (valores[n] - b) / (mm - b)
        cerca += t > 0.5
        filas += (f'<div class="fila-dot"><span class="nombre">{n}</span><div class="pista-dot">'
                  f'<i class="marca-ref b" style="left:{pos(0)}%"></i><i class="marca-ref m" style="left:{pos(1)}%"></i>'
                  f'<i class="punto" style="left:{pos(t):.1f}%" title="{n}: {fmt_num(valores[n])}"></i></div></div>')
    return limpiar(f"""
    <div class="tarjeta salida">
      <div class="veredicto {'m' if es_m else 'b'}"><span class="tag">Resultado del modelo</span><h3>{res['diagnostico']}</h3>
        {medidor(res['probabilidad_maligno'], es_m)}
        <p class="sub">El modelo está {pct(conf)} seguro de que es <b>{res['diagnostico'].lower()}</b>.</p></div>
      <div class="comparacion"><h4>¿Cómo se compara con los casos reales?</h4>{filas}
        <div class="leyenda-dot"><span><i style="background:var(--menta)"></i>promedio benigno</span>
        <span><i style="background:var(--baya)"></i>promedio maligno</span>
        <span><i style="background:var(--rosa-500);border-radius:50%;width:10px;height:10px"></i>tu caso</span></div>
        <p class="descargo"><b>{cerca} de {len(PRINCIPALES)}</b> medidas principales están más cerca del promedio de los malignos.</p></div>
      <p class="descargo">Resultado académico. No reemplaza un diagnóstico médico.</p>
    </div>""")


# ------------------------------------------------------------------ dona del dataset
def dona(res):
    c = 2 * math.pi * 70
    fb = res["benignos"] / res["total"]
    return limpiar(f"""
    <div class="dona">
      <svg viewBox="0 0 200 200" width="200" height="200"><circle cx="100" cy="100" r="70" fill="none" stroke="#e8628a" stroke-width="26"/>
      <circle cx="100" cy="100" r="70" fill="none" stroke="#4fae91" stroke-width="26" stroke-dasharray="{c * fb:.1f} {c:.1f}" transform="rotate(-90 100 100)"/>
      <text x="100" y="98" text-anchor="middle" font-family="Playfair Display,serif" font-weight="700" font-size="34" fill="#3f2136">{res['total']}</text>
      <text x="100" y="120" text-anchor="middle" font-size="12" fill="#6b4560">casos</text></svg>
      <div class="leyenda-dot" style="flex-direction:column;gap:4px;align-items:center">
        <span><i style="background:#4fae91;border-radius:50%;width:10px;height:10px"></i>{res['benignos']} benignos ({pct(fb)})</span>
        <span><i style="background:#e8628a;border-radius:50%;width:10px;height:10px"></i>{res['malignos']} malignos ({pct(1 - fb)})</span></div>
    </div>""")

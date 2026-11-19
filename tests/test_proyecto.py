from fastapi.testclient import TestClient

from backend.main import app
from ml import entrenamiento as ml

cliente = TestClient(app)


def test_dataset_cargado():
    X, y = ml.cargar_datos()
    assert X.shape == (569, 30)
    assert set(y.unique()) == {0, 1}
    assert X.isna().sum().sum() == 0


def test_todos_los_modelos_entrenan():
    for m in ml.MODELOS_DISPONIBLES:
        r = ml.entrenar(m, guardar=False)
        assert r["accuracy"] > 0.9


def test_api_flujo_completo():
    assert cliente.get("/salud").status_code == 200
    r = cliente.post("/entrenar", json={"modelo": "svm", "parametros": {"C": 2.0}, "test_size": 0.25})
    assert r.status_code == 200 and r.json()["accuracy"] > 0.9
    assert cliente.get("/metricas").status_code == 200
    maligno = cliente.get("/ejemplo/maligno").json()
    benigno = cliente.get("/ejemplo/benigno").json()
    assert cliente.post("/predecir", json={"datos": maligno}).json()["diagnostico"] == "Maligno"
    assert cliente.post("/predecir", json={"datos": benigno}).json()["diagnostico"] == "Benigno"


def test_errores():
    assert cliente.post("/entrenar", json={"modelo": "xyz"}).status_code == 400
    assert cliente.post("/predecir", json={"datos": {"mean_radius": 1.0}}).status_code == 422


def test_comparar_y_resumen():
    r = cliente.post("/comparar", json={}).json()
    assert len(r) == 4 and all(m["accuracy"] > 0.9 for m in r)
    s = cliente.get("/resumen").json()
    assert s["total"] == 569 and s["benignos"] + s["malignos"] == 569


def test_graficos_del_frontend():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "frontend"))
    import graficos as g
    m = ml.entrenar("random_forest", guardar=False)
    h = g.resultados(m)
    assert "<svg" in h and "Random Forest" in h and "\n" not in h
    comp = cliente.post("/comparar", json={}).json()
    assert "Aciertos" in g.comparacion(comp)

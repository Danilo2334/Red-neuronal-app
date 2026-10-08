r"""Empaqueta el modelo real del proyecto para Streamlit, sin entrenar.

Desde laboratorio_ia, con energia_streamlit como subcarpeta:
    .\.venv\Scripts\python.exe .\energia_streamlit\preparar_despliegue.py
Despues:
    .\.venv\Scripts\python.exe -m streamlit run .\energia_streamlit\app.py

No publica ni envia archivos: prepara esta carpeta para subirla a GitHub.
"""

import argparse
import hashlib
from importlib import metadata
import json
import platform
from pathlib import Path
import shutil
from zipfile import ZipFile, ZIP_DEFLATED

import joblib
import numpy as np
import pandas as pd

from inferencia import leer_csv, preparar_ventana


def leer_json(ruta):
    return json.loads(ruta.read_text(encoding="utf-8-sig"))


def guardar_json(ruta, contenido):
    ruta.write_text(json.dumps(contenido, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def sha256(ruta):
    digest = hashlib.sha256()
    with ruta.open("rb") as fuente:
        for bloque in iter(lambda: fuente.read(1024 * 1024), b""):
            digest.update(bloque)
    return digest.hexdigest()


def elegir_modelo(proyecto, carpeta=None):
    rutas = [carpeta if carpeta.is_absolute() else proyecto / carpeta] if carpeta else sorted(
        (proyecto / "resultados").glob("lstm_base_12h_disponible_s42_*")
    )
    candidatos = []
    for ruta in rutas:
        if not (ruta / "resumen.json").is_file():
            continue
        datos = leer_json(ruta / "resumen.json")
        if (datos.get("arquitectura") == "Input -> LSTM(64) -> Dense(1, lineal)"
                and datos.get("ventana_horas") == 12 and datos.get("semilla") == 42
                and datos.get("entrenamiento_comun") is False):
            rmse = float(datos["metricas_validacion_lstm"]["RMSE"])
            if not np.isfinite(rmse):
                raise ValueError(f"Validacion no finita: {ruta}")
            candidatos.append((rmse, ruta, datos))
    if not candidatos:
        raise FileNotFoundError("No se encontro A/12h/disponible/semilla42 en resultados. Usa --carpeta-modelo si cambiaste su nombre.")
    _, ruta, datos = min(candidatos, key=lambda item: (item[0], str(item[1])))
    return ruta, datos


def tabla_validacion(proyecto, hash_prepro):
    seleccion = {}
    for ruta in sorted((proyecto / "resultados").glob("*/resumen.json")):
        datos = leer_json(ruta)
        if (datos.get("sha256_preprocesamiento") != hash_prepro
                or datos.get("semilla") != 42 or datos.get("entrenamiento_comun") is not False):
            continue
        nombre = ruta.parent.name
        if nombre.startswith("lstm_base_"):
            modelo = "A: Base"
        elif nombre.startswith("lstm_profunda_"):
            modelo = "B: Profunda"
        elif nombre.startswith("lstm_regularizada_"):
            modelo = "C: Dropout"
        else:
            continue
        ventana = datos.get("ventana_horas")
        if ventana not in (12, 24, 48):
            continue
        m = datos["metricas_validacion_lstm"]
        fila = {"Modelo": modelo, "Ventana (h)": ventana,
                "MAE (MW)": m["MAE"], "MSE (MW²)": m["MSE"], "RMSE (MW)": m["RMSE"],
                "MAPE (%)": m["MAPE_pct"], "R²": m["R2"], "Mejor época": datos["mejor_epoca"]}
        clave = (modelo, ventana)
        if clave not in seleccion or fila["RMSE (MW)"] < seleccion[clave]["RMSE (MW)"]:
            seleccion[clave] = fila
    return sorted(seleccion.values(), key=lambda item: item["RMSE (MW)"])


def exportar(args):
    proyecto = args.proyecto.resolve()
    destino = Path(__file__).resolve().parent
    carpeta, entrenamiento = elegir_modelo(proyecto, args.carpeta_modelo)
    modelo = carpeta / "modelo.keras"
    archivo_prepro = carpeta / "preprocesamiento.joblib"
    archivo_npz = proyecto / "datos" / "preparados" / "ventana_12h.npz"
    prueba = leer_json(carpeta / "prueba_final" / "resumen_prueba.json")
    modelo_hash = sha256(modelo)
    if not prueba.get("prueba_evaluada") or prueba.get("n_test") != 1271:
        raise ValueError("Falta la evaluacion final esperada. Ejecuta primero evaluar_lstm_final.py.")
    if prueba["sha256_modelo"] != modelo_hash:
        raise ValueError("El modelo no coincide con el que se evaluo en prueba.")
    for ruta, clave in [(archivo_prepro, "sha256_preprocesamiento"), (archivo_npz, "sha256_secuencias")]:
        if sha256(ruta) != entrenamiento[clave] or prueba[clave] != entrenamiento[clave]:
            raise ValueError(f"Los archivos no corresponden al entrenamiento y prueba: {ruta}")

    prepro = joblib.load(archivo_prepro)
    escaladores = {
        "columnas_entrada": list(prepro["columnas_entrada"]),
        "columnas_numericas": list(prepro["columnas_numericas"]),
        "categorias_periodo": list(prepro["categorias_periodo"]),
    }
    for nombre in ("scaler_x", "scaler_y"):
        scaler = prepro[nombre]
        if scaler.clip or tuple(scaler.feature_range) != (0, 1):
            raise ValueError("Se esperaba MinMaxScaler con rango [0,1] y sin recorte.")
        escaladores[nombre] = {"scale": scaler.scale_.tolist(), "min": scaler.min_.tolist()}

    with np.load(archivo_npz, allow_pickle=False) as datos:
        final_ejemplo = pd.Timestamp(datos["timestamp_entrada_train"][-1])
        entrada_referencia = datos["X_train"][-1:].copy()
    crudo = leer_csv(proyecto / "datos" / "dataset_energia_depurado.csv")
    fechas = pd.to_datetime(crudo["timestamp"], errors="raise")
    crudo = crudo.loc[fechas.between(final_ejemplo - pd.Timedelta(hours=11), final_ejemplo)].copy()
    columnas_csv = ["timestamp", "demanda_mw", "temperatura_c", "humedad_pct", "viento_kmh",
                    "radiacion_wm2", "precipitacion_mm", "precio_kwh", "festivo"]
    ejemplo = crudo[columnas_csv].copy()
    X_ejemplo, _, _ = preparar_ventana(ejemplo, escaladores)
    if not np.allclose(X_ejemplo, entrada_referencia, atol=2e-7, rtol=0):
        raise ValueError("La preparacion de la app difiere de las secuencias de entrenamiento. No se exporto el paquete.")

    # Fijar las versiones realmente instaladas, en vez de adivinar el entorno.
    versiones = {nombre: metadata.version(nombre) for nombre in ["streamlit", "numpy", "pandas", "keras"]}
    tf_version = entrenamiento["tensorflow_version"]
    instalaciones_tf = []
    for nombre in ["tensorflow", "tensorflow-cpu", "tensorflow-intel"]:
        try:
            instalaciones_tf.append(metadata.version(nombre))
        except metadata.PackageNotFoundError:
            pass
    if tf_version not in instalaciones_tf:
        raise ValueError("La version de TensorFlow del entorno no coincide con la del modelo. Usa la misma .venv del entrenamiento.")
    requisitos = [f"{nombre}=={version}" for nombre, version in versiones.items()]
    # Linux (Streamlit Cloud): paquete CPU. Windows local: TensorFlow ya instalado.
    requisitos += [f'tensorflow-cpu=={tf_version}; platform_system == "Linux"',
                   f'tensorflow=={tf_version}; platform_system != "Linux"']

    resultados = {clave: prueba[clave] for clave in [
        "modelo", "ventana_horas", "n_test", "mejor_epoca", "metricas_validacion_lstm",
        "metricas_prueba_lstm", "metricas_prueba_persistencia", "sha256_modelo",
    ]}
    resultados.update({
        "comparacion_validacion": tabla_validacion(proyecto, entrenamiento["sha256_preprocesamiento"]),
        "python_version": f"{platform.python_version_tuple()[0]}.{platform.python_version_tuple()[1]}",
        "ejemplo": "12 horas historicas de entrenamiento; no son mediciones actuales.",
        "horizonte_horas": 1,
    })
    artefactos = destino / "artefactos"
    artefactos.mkdir(exist_ok=True)
    shutil.copy2(modelo, artefactos / "modelo.keras")
    guardar_json(artefactos / "escaladores.json", escaladores)
    guardar_json(artefactos / "resultados.json", resultados)
    ejemplo.to_csv(artefactos / "ejemplo_12h.csv", index=False, encoding="utf-8-sig")
    (destino / "requirements.txt").write_text("\n".join(requisitos) + "\n", encoding="utf-8")
    (destino / "python_version.txt").write_text(resultados["python_version"] + "\n", encoding="utf-8")
    paquete_listo = destino.parent / "energia_streamlit_lista.zip"
    nombres = ["app.py", "inferencia.py", "preparar_despliegue.py", "requirements.txt",
               "python_version.txt", "LEEME.md", ".gitignore", ".streamlit/config.toml",
               "artefactos/modelo.keras", "artefactos/escaladores.json",
               "artefactos/resultados.json", "artefactos/ejemplo_12h.csv"]
    with ZipFile(paquete_listo, "w", compression=ZIP_DEFLATED) as paquete:
        for nombre in nombres:
            paquete.write(destino / nombre, nombre)
    print("\n=== PAQUETE STREAMLIT PREPARADO ===")
    print(f"Modelo: {carpeta.name}")
    print("Preprocesamiento verificado contra las secuencias originales.")
    print(f"Archivos listos: {destino}")
    print(f"ZIP completo para publicar o compartir: {paquete_listo}")
    print(f"Python que debes elegir en Streamlit Cloud: {resultados['python_version']}")
    print(r"Ejecuta: .\.venv\Scripts\python.exe -m streamlit run .\energia_streamlit\app.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proyecto", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--carpeta-modelo", type=Path, default=None)
    exportar(parser.parse_args())

"""Interfaz de demostración de la LSTM seleccionada: 12 horas -> siguiente hora."""

from __future__ import annotations

import io
import json
import threading
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from inferencia import desescalar_prediccion, leer_csv, preparar_ventana


RAIZ = Path(__file__).resolve().parent
ARTEFACTOS = RAIZ / "artefactos"

st.set_page_config(
    page_title="Demanda de energía · LSTM",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


def numero(valor: object, decimales: int = 2) -> str:
    """Presentación en español, sin convertir faltantes en ceros."""
    try:
        valor = float(valor)
    except (TypeError, ValueError):
        return "—"
    if not np.isfinite(valor):
        return "—"
    return f"{valor:,.{decimales}f}".replace(",", "_").replace(".", ",").replace("_", ".")


def diagnostico(exc: Exception) -> None:
    with st.expander("Detalle para corregir el problema"):
        st.code(f"{type(exc).__name__}: {exc}", language=None)


@st.cache_resource(show_spinner="Cargando el modelo entrenado…")
def cargar_modelo(ruta: str, modificado_ns: int, tamano: int):
    """El sello de archivo invalida la caché cuando cambia el modelo."""
    import tensorflow as tf

    modelo = tf.keras.models.load_model(ruta, compile=False)
    if tuple(modelo.input_shape[1:]) != (12, 16):
        raise ValueError("Se esperaba el modelo seleccionado con entrada de 12 × 16.")
    if tuple(modelo.output_shape[1:]) != (1,):
        raise ValueError("El modelo debe devolver una sola demanda para la siguiente hora.")
    return modelo, threading.Lock()


def tabla_metricas(resultados: dict) -> pd.DataFrame:
    especificaciones = [
        ("MAE (MW)", "MAE", 2),
        ("MSE (MW²)", "MSE", 2),
        ("RMSE (MW)", "RMSE", 2),
        ("MAPE (%)", "MAPE_pct", 2),
        ("R²", "R2", 3),
    ]
    series = {
        "LSTM · validación": resultados.get("metricas_validacion_lstm", {}),
        "LSTM · prueba": resultados.get("metricas_prueba_lstm", {}),
        "Persistencia · prueba": resultados.get("metricas_prueba_persistencia", {}),
    }
    return pd.DataFrame(
        [
            {"Métrica": etiqueta, **{
                nombre: numero(valores.get(clave), decimales)
                for nombre, valores in series.items()
            }}
            for etiqueta, clave, decimales in especificaciones
        ]
    )


def mostrar_resultados(resultados: dict) -> None:
    st.subheader("Lo que mostró el experimento")
    st.write(
        "Se compararon tres arquitecturas con ventanas de 12, 24 y 48 horas. "
        "La LSTM base de 12 horas se seleccionó por su RMSE de validación; "
        "después se evaluó una vez en el periodo reservado para prueba."
    )
    st.dataframe(tabla_metricas(resultados), hide_index=True, width="stretch")

    rmse_lstm = resultados.get("metricas_prueba_lstm", {}).get("RMSE")
    rmse_persistencia = resultados.get("metricas_prueba_persistencia", {}).get("RMSE")
    if isinstance(rmse_lstm, (int, float)) and isinstance(rmse_persistencia, (int, float)):
        if np.isfinite(rmse_lstm) and np.isfinite(rmse_persistencia):
            if rmse_lstm > rmse_persistencia:
                diferencia = (
                    f" El RMSE de la LSTM fue un {numero(100 * (rmse_lstm / rmse_persistencia - 1), 1)} % mayor."
                    if rmse_persistencia > 0 else ""
                )
                st.warning(
                    "En prueba, persistencia obtuvo un RMSE menor que la LSTM."
                    + diferencia
                    + " La LSTM no demostró una mejora frente a esta referencia."
                )
            elif rmse_lstm < rmse_persistencia:
                st.info("En este conjunto de prueba, la LSTM obtuvo un RMSE menor que persistencia.")
            else:
                st.info("Ambos métodos obtuvieron el mismo RMSE de prueba.")

    n_test = resultados.get("n_test")
    st.caption(
        (f"Evaluación sobre {numero(n_test, 0)} muestras de prueba. " if n_test is not None else "")
        + "Las métricas describen el conjunto evaluado: no representan un margen de error garantizado "
        "para cada predicción."
    )
    comparacion = resultados.get("comparacion_validacion")
    if isinstance(comparacion, list) and comparacion:
        with st.expander("Ver las nueve combinaciones"):
            st.dataframe(pd.DataFrame(comparacion), hide_index=True, width="stretch")

    st.markdown(
        "**Alcance de esta demostración**\n\n"
        "- Predice una hora a partir de 12 horas consecutivas ya observadas.\n"
        "- Usa ventanas completas: las métricas no cubren las horas descartadas por datos faltantes.\n"
        "- Entre ventanas cambió también la cantidad de muestras de entrenamiento.\n"
        "- Es un ejercicio académico; sus resultados no acreditan su uso para operar una red eléctrica."
    )


def mostrar_prediccion(historial: pd.DataFrame, prediccion: float, es_ejemplo: bool) -> None:
    ultima_hora = pd.Timestamp(historial.iloc[-1]["timestamp"])
    siguiente_hora = ultima_hora + pd.Timedelta(hours=1)
    persistencia = float(historial.iloc[-1]["demanda_mw"])
    st.divider()
    st.subheader(f"Pronóstico · {siguiente_hora:%d/%m/%Y %H:%M}")
    if es_ejemplo:
        st.caption("Demostración histórica: esta fecha pertenece al ejemplo, no a una predicción en tiempo real.")
    columnas = st.columns(3)
    columnas[0].metric("Última demanda observada", f"{numero(persistencia)} MW")
    columnas[1].metric("Predicción LSTM", f"{numero(prediccion)} MW")
    columnas[2].metric("Referencia: persistencia", f"{numero(persistencia)} MW")
    st.caption(
        "Persistencia repite la última demanda observada. MW significa megavatios y mide potencia. "
        "Las dos predicciones corresponden a la misma hora."
    )

    puntos = historial[["timestamp", "demanda_mw"]].copy()
    puntos = puntos.rename(columns={"timestamp": "Hora", "demanda_mw": "Demanda observada"})
    puntos["LSTM"] = np.nan
    puntos["Persistencia"] = np.nan
    pronostico = pd.DataFrame({
        "Hora": [siguiente_hora],
        "Demanda observada": [np.nan],
        "LSTM": [prediccion],
        "Persistencia": [persistencia],
    })
    puntos = pd.concat([puntos, pronostico], ignore_index=True)
    st.scatter_chart(
        puntos,
        x="Hora",
        y=["Demanda observada", "LSTM", "Persistencia"],
        y_label="Demanda (MW)",
        color=["#64748b", "#0f766e", "#d97706"],
        size=100,
        height=340,
    )
    st.caption("Los últimos dos puntos son pronósticos. No se muestra una demanda real para esa hora.")

    salida = pd.DataFrame({
        "timestamp_ultima_observacion": [ultima_hora.strftime("%Y-%m-%d %H:%M:%S")],
        "timestamp_objetivo": [siguiente_hora.strftime("%Y-%m-%d %H:%M:%S")],
        "prediccion_lstm_mw": [prediccion],
        "prediccion_persistencia_mw": [persistencia],
        "ventana_horas": [12],
        "fuente": ["ejemplo_historico" if es_ejemplo else "csv_cargado"],
    })
    st.download_button(
        "Descargar predicción en CSV",
        data=salida.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"prediccion_energia_{siguiente_hora:%Y%m%d_%H%M}.csv",
        mime="text/csv",
        on_click="ignore",
    )


def mostrar_simulador(escaladores: dict) -> None:
    st.subheader("Carga el historial y predice la siguiente hora")
    st.write(
        "Usaremos las últimas 12 horas del archivo, ordenadas por fecha. "
        "Deben ser consecutivas y tener todas las variables de entrada completas."
    )
    fuente = st.radio(
        "Origen de los datos",
        ["Ejemplo histórico", "Cargar mi CSV"],
        horizontal=True,
    )
    es_ejemplo = fuente == "Ejemplo histórico"
    if es_ejemplo:
        ruta_ejemplo = ARTEFACTOS / "ejemplo_12h.csv"
        if not ruta_ejemplo.is_file():
            st.info("No hay ejemplo disponible. Puedes elegir «Cargar mi CSV».")
            return
        contenido = ruta_ejemplo.read_bytes()
        st.info("Este ejemplo contiene observaciones históricas reales del dataset del taller.")
        st.download_button(
            "Descargar ejemplo de 12 horas",
            contenido,
            file_name="ejemplo_12h.csv",
            mime="text/csv",
            on_click="ignore",
        )
    else:
        archivo = st.file_uploader("Archivo CSV con el historial horario", type=["csv"])
        st.caption(
            "Incluye timestamp, demanda_mw, temperatura_c, humedad_pct, viento_kmh, radiacion_wm2, "
            "precipitacion_mm, precio_kwh y festivo. El calendario se obtiene de las fechas. "
            "No necesitas incluir la demanda futura. Usa punto como separador decimal."
        )
        if archivo is None:
            return
        contenido = archivo.getvalue()

    try:
        datos = leer_csv(io.BytesIO(contenido))
        entrada, historial, avisos = preparar_ventana(datos, escaladores)
    except (ValueError, TypeError, KeyError, UnicodeError, pd.errors.ParserError) as exc:
        st.error(f"No se puede usar este historial: {exc}")
        return
    except Exception as exc:
        st.error("No se pudo leer o preparar el archivo.")
        diagnostico(exc)
        return

    for aviso in avisos:
        st.warning(str(aviso))
    inicio = pd.Timestamp(historial.iloc[0]["timestamp"])
    fin = pd.Timestamp(historial.iloc[-1]["timestamp"])
    st.success(f"12 horas listas · {inicio:%d/%m/%Y %H:%M} a {fin:%d/%m/%Y %H:%M}")
    with st.expander("Revisar las 12 observaciones", expanded=False):
        st.dataframe(historial, hide_index=True, width="stretch")

    if st.button("Predecir siguiente hora", type="primary"):
        try:
            ruta_modelo = ARTEFACTOS / "modelo.keras"
            sello = ruta_modelo.stat()
            modelo, bloqueo = cargar_modelo(str(ruta_modelo), sello.st_mtime_ns, sello.st_size)
            with st.spinner("Calculando la demanda…"):
                with bloqueo:
                    normalizada = modelo(entrada, training=False).numpy()
                prediccion = desescalar_prediccion(normalizada, escaladores)
            if not np.isfinite(prediccion):
                raise ValueError("El modelo produjo una demanda no finita.")
        except Exception as exc:
            st.error("No fue posible generar la predicción con el modelo guardado.")
            diagnostico(exc)
            return
        mostrar_prediccion(historial, float(prediccion), es_ejemplo)


def main() -> None:
    st.title("⚡ Demanda de energía")
    st.write("De las últimas **12 horas** a la demanda de la **siguiente hora**.")
    with st.sidebar:
        st.subheader("Laboratorio LSTM")
        st.write("**Modelo seleccionado**\n\nLSTM base · ventana de 12 horas")
        st.divider()
        st.write("**16 variables por hora**\n\nDemanda, clima, precio y calendario.")
        st.caption("Selección en validación · evaluación final en prueba")

    requeridos = [ARTEFACTOS / nombre for nombre in ("modelo.keras", "escaladores.json", "resultados.json")]
    if any(not ruta.is_file() for ruta in requeridos):
        st.warning("Falta preparar el modelo. Ejecuta preparar_despliegue.py.")
        st.stop()
    try:
        escaladores = json.loads((ARTEFACTOS / "escaladores.json").read_text(encoding="utf-8"))
        resultados = json.loads((ARTEFACTOS / "resultados.json").read_text(encoding="utf-8"))
        if not isinstance(escaladores, dict) or not isinstance(resultados, dict):
            raise ValueError("Los archivos de configuración deben contener objetos JSON.")
    except Exception as exc:
        st.error("No se pudo cargar la configuración del modelo.")
        diagnostico(exc)
        st.stop()

    simulador, experimento = st.tabs(["Predecir demanda", "Resultados del experimento"])
    with simulador:
        mostrar_simulador(escaladores)
    with experimento:
        mostrar_resultados(resultados)


if __name__ == "__main__":
    main()

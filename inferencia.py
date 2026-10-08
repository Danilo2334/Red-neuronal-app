"""Preparación de entradas e inversión del escalado, sin TensorFlow ni Streamlit."""

from __future__ import annotations

import csv
import io
from pathlib import Path

import numpy as np
import pandas as pd


HORAS = 12
NUMERICAS = [
    "demanda_mw", "temperatura_c", "humedad_pct", "viento_kmh",
    "radiacion_wm2", "precipitacion_mm", "precio_kwh", "hora",
    "dia_semana", "fin_semana", "festivo", "mes",
]
PERIODOS = ["madrugada", "mañana", "tarde", "noche"]
MEDICIONES = [
    "demanda_mw", "temperatura_c", "humedad_pct", "viento_kmh",
    "radiacion_wm2", "precipitacion_mm", "precio_kwh", "festivo",
]
ENTRADAS = NUMERICAS + [f"periodo_{categoria}" for categoria in PERIODOS]


def leer_csv(origen) -> pd.DataFrame:
    """Lee una ruta, bytes o archivo abierto; admite coma, punto y coma y tabulación."""
    if isinstance(origen, (str, Path)):
        contenido = Path(origen).read_bytes()
    elif isinstance(origen, (bytes, bytearray)):
        contenido = bytes(origen)
    elif hasattr(origen, "getvalue"):
        contenido = origen.getvalue()
    elif hasattr(origen, "read"):
        posicion = None
        try:
            posicion = origen.tell()
            origen.seek(0)
        except (AttributeError, OSError, io.UnsupportedOperation):
            pass
        contenido = origen.read()
        if posicion is not None:
            try:
                origen.seek(posicion)
            except (AttributeError, OSError, io.UnsupportedOperation):
                pass
    else:
        raise ValueError("Entrega un archivo CSV o una ruta válida.")

    if isinstance(contenido, bytes):
        try:
            texto = contenido.decode("utf-8-sig")
        except UnicodeDecodeError:
            try:
                texto = contenido.decode("cp1252")
            except UnicodeDecodeError as error:
                raise ValueError("No se pudo leer la codificación del CSV.") from error
    else:
        texto = str(contenido).lstrip("\ufeff")
    if not texto.strip():
        raise ValueError("El archivo CSV está vacío.")
    try:
        separador = csv.Sniffer().sniff(texto[:65536], delimiters=",;\t|").delimiter
    except csv.Error:
        cabecera = next((linea for linea in texto.splitlines() if linea.strip()), "")
        separador = max(",;\t|", key=cabecera.count)
        if not cabecera.count(separador):
            raise ValueError("No se reconoce el separador del CSV.") from None
    cabeceras = next((fila for fila in csv.reader(io.StringIO(texto), delimiter=separador) if fila), [])
    nombres = [nombre.strip() for nombre in cabeceras]
    if len(nombres) != len(set(nombres)):
        raise ValueError("El CSV contiene nombres de columnas repetidos.")
    try:
        # Excel puede usar coma decimal cuando el separador es punto y coma.
        df = pd.read_csv(io.StringIO(texto), sep=separador,
                         decimal="," if separador == ";" else ".")
    except (pd.errors.ParserError, pd.errors.EmptyDataError, ValueError) as error:
        raise ValueError(f"No se pudo interpretar el CSV: {error}") from error
    df.columns = [str(nombre).strip().lstrip("\ufeff") for nombre in df.columns]
    if df.columns.duplicated().any():
        raise ValueError("El CSV contiene nombres de columnas repetidos.")
    return df


def _coeficientes(escaladores: dict, clave: str, cantidad: int):
    try:
        bloque = escaladores[clave]
        escala = np.asarray(bloque["scale"], dtype=np.float64)
        minimo = np.asarray(bloque["min"], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"El archivo de escaladores no contiene {clave} válido.") from error
    if (escala.shape != (cantidad,) or minimo.shape != (cantidad,)
            or not np.isfinite(escala).all() or not np.isfinite(minimo).all()
            or (escala <= 0).any()):
        raise ValueError(f"Los coeficientes de {clave} son inválidos.")
    return escala, minimo


def preparar_ventana(df: pd.DataFrame, escaladores: dict):
    """Usa exclusivamente las últimas 12 horas; rechaza datos incompletos.

    Devuelve (entrada normalizada de forma (1, 12, 16), historial, avisos).
    Las fechas representan horas locales sin zona horaria. No usa objetivos.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Se esperaba una tabla de datos.")
    if df.columns.duplicated().any():
        raise ValueError("La tabla contiene nombres de columnas repetidos.")
    faltantes = [nombre for nombre in ["timestamp"] + MEDICIONES if nombre not in df.columns]
    if faltantes:
        raise ValueError("Faltan columnas obligatorias: " + ", ".join(faltantes) + ".")
    if len(df) < HORAS:
        raise ValueError(f"Se necesitan al menos {HORAS} horas; se recibieron {len(df)} filas.")
    if (escaladores.get("columnas_numericas") != NUMERICAS
            or escaladores.get("categorias_periodo") != PERIODOS
            or escaladores.get("columnas_entrada") != ENTRADAS):
        raise ValueError("Las variables y su orden no coinciden con el modelo de 16 entradas.")
    escala, minimo = _coeficientes(escaladores, "scaler_x", len(ENTRADAS))

    fechas = []
    for numero, valor in enumerate(df["timestamp"], start=1):
        try:
            if isinstance(valor, (bool, int, float, np.number)) or pd.isna(valor):
                raise ValueError()
            fecha = pd.Timestamp(valor)
            if pd.isna(fecha):
                raise ValueError()
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"Timestamp inválido en la fila de datos {numero}.") from error
        if fecha.tzinfo is not None:
            raise ValueError("Los timestamps deben ser horas locales sin zona horaria; no se convierten automáticamente.")
        fechas.append(fecha)
    tabla = df.copy()
    tabla["timestamp"] = pd.to_datetime(fechas)
    if tabla["timestamp"].duplicated().any():
        repetida = tabla.loc[tabla["timestamp"].duplicated(), "timestamp"].iloc[0]
        raise ValueError(f"Timestamp duplicado: {repetida}. Revisa el CSV; no se promedian ni eliminan duplicados.")

    avisos = []
    if not tabla["timestamp"].is_monotonic_increasing:
        avisos.append("Las filas se ordenaron cronológicamente antes de tomar las últimas 12 horas.")
    if len(tabla) > HORAS:
        avisos.append(f"Se usan las últimas 12 horas de las {len(tabla)} filas recibidas.")
    ventana = tabla.sort_values("timestamp").tail(HORAS).reset_index(drop=True)
    fechas_finales = ventana["timestamp"]
    if not fechas_finales.eq(fechas_finales.dt.floor("h")).all():
        raise ValueError("Las últimas 12 observaciones deben corresponder a horas exactas, sin minutos ni segundos.")
    if not fechas_finales.diff().iloc[1:].eq(pd.Timedelta(hours=1)).all():
        raise ValueError("Hay huecos en las últimas 12 horas. No se reemplazan por una ventana anterior.")

    for columna in MEDICIONES:
        fuente = ventana[columna]
        if columna == "festivo":
            fuente = fuente.map(lambda valor: {"true": 1, "false": 0}.get(valor.strip().lower(), valor)
                                if isinstance(valor, str) else valor)
        valores = pd.to_numeric(fuente, errors="coerce").astype(float)
        if not np.isfinite(valores.to_numpy()).all():
            raise ValueError(f"La columna {columna} tiene datos faltantes, no numéricos o infinitos en las últimas 12 horas. No se imputan valores.")
        ventana[columna] = valores
    if not ventana["festivo"].isin([0, 1]).all():
        raise ValueError("festivo debe contener únicamente 0/1 o valores booleanos. Se conserva el calendario de la fuente.")
    if not ventana["humedad_pct"].between(0, 100).all():
        raise ValueError("humedad_pct debe estar entre 0 y 100 en las últimas 12 horas.")
    for columna in ["demanda_mw", "viento_kmh", "radiacion_wm2", "precipitacion_mm"]:
        if ventana[columna].lt(0).any():
            raise ValueError(f"{columna} no puede ser negativa en las últimas 12 horas.")
    if ventana["precio_kwh"].lt(0).any():
        avisos.append("Hay precios negativos. Se conservan; su validez depende del mercado y de la definición de la fuente.")

    derivados = {
        "hora": fechas_finales.dt.hour,
        "dia_semana": fechas_finales.dt.dayofweek,
        "fin_semana": fechas_finales.dt.dayofweek.ge(5).astype(int),
        "mes": fechas_finales.dt.month,
    }
    corregidas = []
    for columna, valores in derivados.items():
        if columna in ventana and not pd.to_numeric(ventana[columna], errors="coerce").eq(valores).all():
            corregidas.append(columna)
        ventana[columna] = valores
    periodo = ventana["hora"].map(lambda hora: PERIODOS[int(hora) // 6])
    if "periodo_dia" in ventana and not ventana["periodo_dia"].astype("string").eq(periodo).fillna(False).all():
        corregidas.append("periodo_dia")
    ventana["periodo_dia"] = periodo
    if corregidas:
        avisos.append("Se corrigieron desde timestamp: " + ", ".join(corregidas) + ".")

    entradas = ventana[NUMERICAS].astype(np.float64).copy()
    for categoria in PERIODOS:
        entradas[f"periodo_{categoria}"] = periodo.eq(categoria).astype(float)
    matriz = entradas[ENTRADAS].to_numpy(dtype=np.float64)
    normalizada = matriz * escala + minimo
    fuera = ((normalizada < -1e-7) | (normalizada > 1 + 1e-7)).any(axis=0)
    if fuera.any():
        nombres = [nombre for nombre, excede in zip(ENTRADAS, fuera) if excede]
        avisos.append("Hay valores fuera del rango aprendido en entrenamiento: "
                      + ", ".join(nombres) + ". Se conservan sin recortar ni reajustar el escalador.")
    with np.errstate(over="ignore", invalid="ignore"):
        entrada = normalizada.astype(np.float32)[None, :, :]
    if not np.isfinite(entrada).all():
        raise ValueError("La magnitud de alguna entrada impide representarla numéricamente para el modelo.")
    historial = ventana[["timestamp"] + NUMERICAS + ["periodo_dia"]].copy()
    return entrada, historial, avisos


def desescalar_prediccion(pred, escaladores: dict) -> float:
    """Invierte MinMaxScaler para una predicción escalar; no limita su valor."""
    escala, minimo = _coeficientes(escaladores, "scaler_y", 1)
    try:
        valor = np.asarray(pred, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("La predicción debe ser un único número.") from error
    if valor.size != 1 or not np.isfinite(valor).all():
        raise ValueError("La predicción debe contener un único número finito.")
    resultado = float((valor.reshape(-1)[0] - minimo[0]) / escala[0])
    if not np.isfinite(resultado):
        raise ValueError("La predicción en MW no es finita.")
    return resultado

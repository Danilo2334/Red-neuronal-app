# Demanda energetica: LSTM y persistencia

App Streamlit del taller: LSTM base de 12 horas, pronostico a una hora.
Incluye una comparacion honesta con persistencia: esta ultima obtuvo mejor
rendimiento en el conjunto final de prueba. La app es una demostracion del
experimento; no afirma superioridad de la LSTM.

## 1. Preparar en tu computador

Extrae `energia_streamlit` dentro de `laboratorio_ia`, junto a `scripts`,
`datos` y `resultados`. Los pesos reales estan en tu equipo: este paquete
inicial incluye el codigo, y el siguiente comando copia tu modelo y sus
parametros de normalizacion.

Desde la terminal PowerShell de `laboratorio_ia`, ejecuta uno por uno:

```powershell
.\.venv\Scripts\python.exe -m pip install streamlit
.\.venv\Scripts\python.exe .\energia_streamlit\preparar_despliegue.py
.\.venv\Scripts\python.exe -m streamlit run .\energia_streamlit\app.py
```

Entra a http://localhost:8501 si no se abre automaticamente. En la app, usa
el ejemplo historico y pulsa **Predecir siguiente hora**. Tambien puedes subir
un CSV propio. Ctrl+C detiene el servidor en la terminal.

El preparador verifica los hashes del modelo evaluado, los escaladores y las
secuencias. Exporta la transformacion MinMax a JSON y comprueba que una ventana
reconstruida coincide con el NPZ original. No entrena ni ajusta escaladores.
No instala paquetes ni modifica los resultados originales.

Tambien crea `energia_streamlit_lista.zip` en `laboratorio_ia`, con la app y
el modelo real. Ese archivo permite compartir el paquete completo sin enviar
el dataset entero ni las demas ejecuciones. Sus archivos van en la raiz del
repositorio cuando lo descomprimas para publicar.

Al finalizar, esta carpeta contiene `app.py`, `inferencia.py`,
`requirements.txt`, `python_version.txt`, `.streamlit/config.toml` y estos
archivos imprescindibles:

- `artefactos/modelo.keras`: el checkpoint real seleccionado y evaluado.
- `artefactos/escaladores.json`: los parametros aprendidos en entrenamiento.
- `artefactos/resultados.json`: metricas reales de tus ejecuciones.
- `artefactos/ejemplo_12h.csv`: una ventana historica de entrenamiento.

## 2. Publicar en Streamlit Community Cloud

1. Crea un repositorio en tu cuenta de GitHub, por ejemplo `energia-lstm`.
2. Sube **el contenido de la carpeta `energia_streamlit` ya preparada** al
   repositorio. Asegurate de incluir los cuatro archivos de `artefactos`.
   No necesitas subir la `.venv`, el dataset completo ni todas las ejecuciones.
3. Abre https://share.streamlit.io/ e inicia sesion con la cuenta de GitHub que
   tiene el repositorio. Selecciona **Create app** y la opcion de app existente.
4. Elige tu repositorio, su rama y `app.py` como archivo principal si subiste
   el contenido a la raiz. Si subiste la carpeta completa, la ruta es
   `energia_streamlit/app.py`.
5. En **Advanced settings**, selecciona la version de Python indicada por
   el preparador y guardada en `python_version.txt`. Ese archivo es una
   referencia: la version se selecciona en la interfaz de Streamlit.
6. Pulsa **Deploy**. Prueba el ejemplo y un CSV en la URL publicada.

El despliegue necesita tu modelo real y una cuenta de Streamlit conectada al
repositorio. Este ZIP, antes de ejecutar el preparador, no contiene pesos y
todavia no es una app publicada.

## Entradas y comportamiento

CSV UTF-8, con punto decimal y encabezados:

`timestamp,demanda_mw,temperatura_c,humedad_pct,viento_kmh,radiacion_wm2,precipitacion_mm,precio_kwh,festivo`

`timestamp` usa `AAAA-MM-DD HH:MM:SS` sin zona horaria, como el dataset del
taller. Todas las horas deben corresponder al mismo reloj local del dataset.
`festivo` debe ser 0 o 1, segun el calendario usado en los datos originales.
Las unidades son las mismas que en el dataset; el precio no se convierte.

Puedes subir el CSV depurado completo: se utilizan sus ultimas 12 horas, que
deben estar completas y ser consecutivas. Si tienen faltantes, corrige la
fuente o sube otra ventana conocida e identificala como historica. La app
no rellena huecos ni busca silenciosamente un periodo anterior.

Hora, dia de semana, fin de semana, mes y periodo del dia se reconstruyen
desde la fecha. `demanda_objetivo` y las columnas sobrantes no se usan como
entradas. No se necesitan datos de la hora futura.

La salida es la demanda de la hora siguiente en MW. Persistencia equivale
a repetir la demanda de la ultima hora. El ejemplo descargable es historico,
no corresponde a datos en vivo. El intervalo de 12 horas es historia de
entrada, no un pronostico de las siguientes 12 horas.

## Verificacion y limites

El exportador verifica la identidad de los artefactos y la igualdad del
preprocesamiento respecto a una secuencia original. La carga y prediccion
con el modelo real deben ejecutarse en tu entorno y luego en el despliegue:
el modelo entrenado no fue adjuntado a esta conversacion.

No se ajusta el modelo a partir de los resultados de prueba. Las metricas
corresponden a ventanas completas conservadas tras la limpieza; no cubren
automaticamente todas las horas ni garantizan el rendimiento futuro.

Si el despliegue falla, abre **Manage app > Logs** y revisa el mensaje. Las
versiones estan fijadas a tu entorno; no cambies TensorFlow al azar. Si falta
un archivo de artefactos, vuelve al paso de preparacion y subelo al repositorio.

Fuentes oficiales consultadas para el despliegue:

- https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy
- https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import streamlit as st
from pathlib import Path

# -----------------------
# Config
# -----------------------
st.set_page_config(page_title="Demo IA Logística", layout="centered")

ARCHIVO_CSV = Path("datos_pedidos.csv")

# -----------------------
# Funciones
# -----------------------
def cargar_datos():
    if not ARCHIVO_CSV.exists():
        st.error("No se encontró datos_pedidos.csv en la carpeta del proyecto.")
        st.stop()

    df = pd.read_csv(ARCHIVO_CSV)

    # Validación mínima de columnas
    columnas_necesarias = {"fecha", "dia_semana", "temperatura", "festivo", "pedidos"}
    if not columnas_necesarias.issubset(set(df.columns)):
        st.error(f"El CSV debe tener estas columnas: {sorted(list(columnas_necesarias))}")
        st.stop()

    return df

def entrenar_modelo(df):
    X = df[["dia_semana", "temperatura", "festivo"]]
    y = df["pedidos"]
    modelo = LinearRegression()
    modelo.fit(X, y)
    return modelo, X, y

def nivel_demanda(pred, media):
    if pred < media * 0.85:
        return "Baja"
    if pred > media * 1.15:
        return "Alta"
    return "Media"

def recomendar_repartidores(pedidos_estimados, capacidad_por_repartidor=25, margen_seguridad=0.10):
    """
    Regla simple (no ML): pedidos / capacidad, con un margen de seguridad.
    Ej: si un repartidor puede gestionar 25 pedidos/día y añadimos 10% margen.
    """
    pedidos_con_margen = pedidos_estimados * (1 + margen_seguridad)
    recomendados = int((pedidos_con_margen + capacidad_por_repartidor - 1) // capacidad_por_repartidor)  # ceil
    return max(recomendados, 1)

# -----------------------
# App
# -----------------------
st.title("IA aplicada a Logística: predicción de pedidos")
st.write(
    "App web en Python con **Machine Learning**. "
    "Carga un histórico (CSV), entrena un modelo y predice la demanda para planificar recursos."
)

with st.expander("¿Qué demuestra esta demo?"):
    st.markdown(
        "- **Big Data (en pequeño)**: histórico de pedidos en un CSV.\n"
        "- **Machine Learning**: modelo entrenado con ese histórico.\n"
        "- **IA aplicada**: predicción + decisión operativa (repartidores)."
    )

df = cargar_datos()
modelo, X, y = entrenar_modelo(df)

media = float(df["pedidos"].mean())
min_pedidos = int(df["pedidos"].min())
max_pedidos = int(df["pedidos"].max())

st.subheader("1) Configuración del cálculo")
dias = {1: "Lunes", 2: "Martes", 3: "Miércoles", 4: "Jueves", 5: "Viernes", 6: "Sábado", 7: "Domingo"}

dia = st.selectbox("Día de la semana", options=list(dias.keys()), format_func=lambda x: dias[x])
temp = st.slider("Temperatura (°C)", min_value=-5, max_value=45, value=18)
festivo = st.toggle("Es festivo", value=False)

st.subheader("2) Parámetros para recomendar repartidores (operativa)")
capacidad = st.number_input("Capacidad por repartidor (pedidos/día)", min_value=5, max_value=200, value=25, step=5)
margen = st.slider("Margen de seguridad (%)", min_value=0, max_value=50, value=10, step=5)

col1, col2, col3 = st.columns(3)
with col1:
    calcular = st.button("Calcular predicción", type="primary")
with col2:
    st.metric("Pedidos típicos (media)", f"{int(round(media))}")
with col3:
    st.metric("Rango histórico", f"{min_pedidos} - {max_pedidos}")

if calcular:
    entrada = pd.DataFrame([[dia, temp, 1 if festivo else 0]], columns=["dia_semana", "temperatura", "festivo"])
    pred = float(modelo.predict(entrada)[0])
    pred_int = int(round(pred))

    st.success(f"Pedidos estimados: **{pred_int}**")
    st.info(f"Nivel de demanda: **{nivel_demanda(pred, media)}**")

    # Recomendación de repartidores
    repartidores = recomendar_repartidores(
        pedidos_estimados=pred_int,
        capacidad_por_repartidor=int(capacidad),
        margen_seguridad=float(margen) / 100.0
    )
    st.subheader("3) Recomendación operativa")
    st.write(
        f"Con una capacidad de **{int(capacidad)} pedidos/repartidor** y un margen de **{int(margen)}%**, "
        f"se recomiendan **{repartidores} repartidores**."
    )

    # Mensaje inteligente según demanda
    if repartidores >= 6:
        st.warning("Alta carga operativa: se recomienda reforzar turnos y preparar rutas adicionales.")
    elif repartidores >= 4:
        st.info("Carga media: planificación estándar con ligera previsión de refuerzo.")
    else:
        st.success("Carga baja: se pueden optimizar turnos y reducir costes operativos.")



    # Comparación con típico
    diff = pred - media
    signo = "+" if diff >= 0 else ""
    st.caption(f"Comparación con día típico: {signo}{int(round(diff))} pedidos aprox.")

    st.subheader("4) Evidencia (para justificar)")
    with st.expander("Ver datos y gráficas"):
        st.write("Histórico cargado (CSV)")
        st.dataframe(df, use_container_width=True)

        st.write("Reales vs predichos (sobre el histórico)")
        y_pred = modelo.predict(X)
        fig = plt.figure()
        plt.scatter(y, y_pred)
        plt.xlabel("Pedidos reales")
        plt.ylabel("Pedidos predichos")
        plt.title("Comparación pedidos reales vs predichos")
        st.pyplot(fig)

        st.write("Influencia de variables (coeficientes del modelo)")
        coefs = pd.DataFrame({"variable": ["dia_semana", "temperatura", "festivo"], "coeficiente": modelo.coef_})
        st.dataframe(coefs, use_container_width=True)

st.divider()
st.caption("Demo educativa: predicción (ML) + decisión (recomendación de recursos) con reglas simples.")

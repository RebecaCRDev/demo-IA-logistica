# IA aplicada a Logística — Predicción de Demanda

Proyecto desarrollado en **Python** como demostración práctica de
**Inteligencia Artificial aplicada a los sectores productivos (DAM)**.

La aplicación utiliza **Machine Learning** para predecir el número de pedidos
a partir de datos históricos y, en base a esa predicción, recomienda el número
de repartidores necesarios para optimizar la operación logística.

---

## Tecnologías utilizadas

- **Python**
- **Pandas** → gestión de datos
- **Scikit-learn** → modelo de Machine Learning (regresión lineal)
- **Matplotlib** → visualización de resultados
- **Streamlit** → aplicación web interactiva

---

## Funcionamiento

1. Se carga un archivo CSV con datos históricos de pedidos:
   - día de la semana
   - temperatura
   - festivo
   - número de pedidos

2. Se entrena un modelo de **Machine Learning** con esos datos.

3. El usuario introduce las condiciones de un día futuro.

4. La aplicación:
   - **predice la demanda de pedidos**
   - **recomienda el número de repartidores**
   - muestra **gráficas y evidencia del modelo**

Esto simula un **caso real de IA aplicada a logística empresarial**.

---

## Estructura del proyecto

```
demo_ia_logistica/
│
├── app.py                # Aplicación web en Streamlit
├── datos_pedidos.csv     # Datos históricos de entrenamiento
├── requirements.txt      # Dependencias del proyecto
└── README.md             # Documentación
```

---

## Cómo ejecutar la aplicación

### 1. Clonar el repositorio

```bash
git clone https://github.com/RebecaCRDev/demo-IA-logistica.git
cd demo-IA-logistica
```

### 2. Crear entorno virtual

```bash
python -m venv .venv
```

Activar:

**Windows**

```bash
.venv\Scripts\activate
```

**Mac / Linux**

```bash
source .venv/bin/activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4. Ejecutar la app

```bash
streamlit run app.py
```

Se abrirá automáticamente en el navegador.

---

## Objetivo académico

Este proyecto demuestra:

- Uso de **datos reales** en formato CSV
- Entrenamiento de un modelo de **Machine Learning**
- Desarrollo de una **aplicación web interactiva en Python**
- Aplicación de la **IA para la toma de decisiones logísticas**

Trabajo realizado para la asignatura
**Digitalización aplicada a los sectores productivos (DAM)**.

---

## Autora

**Rebeca CR**

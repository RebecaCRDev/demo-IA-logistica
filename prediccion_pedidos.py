import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Datos simulados de pedidos
data = {
    "dia_semana": [1, 2, 3, 4, 5, 6, 7, 1, 2, 3, 4, 5, 6, 7],
    "temperatura": [15, 18, 20, 22, 25, 30, 28, 16, 19, 21, 23, 26, 31, 29],
    "festivo": [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1],
    "pedidos": [50, 55, 60, 65, 80, 120, 110, 52, 58, 63, 70, 85, 130, 115]
}

df = pd.DataFrame(data)

# Variables de entrada (X) y salida (y)
X = df[["dia_semana", "temperatura", "festivo"]]
y = df["pedidos"]

# Crear y entrenar modelo
modelo = LinearRegression()
modelo.fit(X, y)

print("Modelo entrenado correctamente.")

# ---- Predicción interactiva ----
print("\nIntroduce datos para predecir pedidos:")

dia = int(input("Día de la semana (1-7): "))
temp = float(input("Temperatura estimada: "))
festivo = int(input("¿Es festivo? (0=no, 1=sí): "))

import pandas as pd
entrada = pd.DataFrame([[dia, temp, festivo]], columns=["dia_semana", "temperatura", "festivo"])
prediccion = modelo.predict(entrada)

print(f"\nPedidos estimados: {int(prediccion[0])}")
print("\nExplicación:")
print("Esta predicción usa Machine Learning entrenado con datos históricos.")
print("Permite a la empresa planificar repartidores y reducir costes logísticos.")

# ---- Gráfica simple ----
plt.scatter(y, modelo.predict(X))
plt.xlabel("Pedidos reales")
plt.ylabel("Pedidos predichos")
plt.title("Comparación pedidos reales vs predichos")
plt.show()

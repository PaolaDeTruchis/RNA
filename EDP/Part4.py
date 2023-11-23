import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, Reshape
from tensorflow.keras.optimizers import RMSprop, Adam

from matplotlib import pyplot as plt
import numpy as np

from PDESolver import PDESolver

# La solucion analitica
def analytical_solution(x):
    return np.sin(x) * (1 - x * np.cos(x)) + x * np.sin(x)

# Generacion de data de entrenamiento
x_train = np.linspace(-5, 5, 1000)

# Creacion del modelo
model = PDESolver()
model.compile(optimizer='adam', loss='mse')

# Entrenamiento del modelo
model.fit(x_train, analytical_solution(x_train), epochs=100, verbose=1)

# Predicciones con el modelo entrenado
x_test = np.linspace(-5, 5, 1000)
predictions = model.predict(x_test)

# Calculo de la solucion analitica
analytical_result = analytical_solution(x_test)

# Affichage des résultats
plt.figure(figsize=(8, 6))
plt.plot(x_test, predictions, label='Solution numérique (NN)', linestyle='--')
plt.plot(x_test, analytical_result, label='Solution analytique', linestyle='-', linewidth=2)
plt.title("Solution numérique vs Solution analytique")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()


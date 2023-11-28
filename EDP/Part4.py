import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, Reshape
from tensorflow.keras.optimizers import RMSprop, Adam

from matplotlib import pyplot as plt
import numpy as np

from new_ODESolver import new_ODESolver


def analytic_solu1 (x):
    return (x/2) * np.sin(x)

x_train1 = np.random.uniform(-5, 5, 1000)  # las valores aleatorias deben estar en el intervalo [-5, 5]
x_train1 = np.sort(x_train1) # sort permite ordenanr las valores de 'x'
y_train1 = analytic_solu1(x_train1)


solver = new_ODESolver()


solver.compile(optimizer='adam')
solver.fit(x_train1, y_train1, epochs=10, batch_size=100)


# Generando datos para evaluación en el intervalo [-1, 1]
x_eval = np.linspace(-5, 5, 1000)
y_pred1 = solver.predict(x_eval)

# Trazar gráficas para la función b
plt.figure(figsize=(8, 6))
plt.scatter(x_train1, y_train1, label='Data')
plt.plot(x_eval, y_pred1, color='red', label='Predictions')
plt.plot(x_eval, analytic_solu1(x_eval), color='green', linestyle='--', label='True function')
plt.title('Approximation de la fonction a')
plt.legend()
plt.show()



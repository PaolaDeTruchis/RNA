import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, Reshape, Conv1D
from tensorflow.keras.optimizers import RMSprop, Adam

from matplotlib import pyplot as plt
import numpy as np

from new_ODESolver import new_ODESolver
from PDESolver import PDESolver


def analytic_solu1 (x):
    return np.cos(x) - 0.5*np.sin(x)

x_train1 = np.random.uniform(-20, 20, 10000)  # las valores aleatorias deben estar en el intervalo [-5, 5]
x_train1 = np.sort(x_train1) # sort permite ordenanr las valores de 'x'
y_train1 = analytic_solu1(x_train1)

x_train1 = np.reshape(x_train1, (-1, 1, 1))
y_train1 = np.reshape(y_train1, (-1, 1, 1))

solver = PDESolver()

solver.summary()

optimizer = Adam(learning_rate = 0.0001)
solver.compile(optimizer=optimizer, loss='mse', metrics=['accuracy'])
solver.fit(x_train1, y_train1, epochs=10, batch_size=100)


# Generando datos para evaluación en el intervalo [-1, 1]
x_eval = np.linspace(-10, 10, 1000)
y_pred1 = solver.predict(x_eval)

# Trazar gráficas para la función b
plt.figure(figsize=(8, 6))
plt.plot(x_eval, y_pred1, color='red', label='Predictions')
plt.plot(x_eval, analytic_solu1(x_eval), color='green', linestyle='--', label='True function')
plt.title('Approximation de la fonction a')
plt.legend()
plt.show()




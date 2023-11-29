import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, Reshape
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.regularizers import L1, L2

from matplotlib import pyplot as plt
import numpy as np

from new_ODESolver import new_ODESolver


def analytic_solu1 (x):
    return x*np.sin(x) - 2*(np.sin(x) - x*np.cos(x))/x

x_train1 = np.random.uniform(-5, 5, 1000)  # las valores aleatorias deben estar en el intervalo [-5, 5]
x_train1 = np.sort(x_train1) # sort permite ordenanr las valores de 'x'

solver = new_ODESolver()
solver.add(Dense(64, activation='tanh', input_shape=(1,)))
solver.add(Dense(356, activation='tanh', kernel_regularizer=L2(0.01)))
solver.add(Dropout(0.2))
solver.add(Dense(128, activation='tanh', kernel_regularizer=L2(0.01)))
solver.add(Dense(1))

solver.summary()

solver.compile(optimizer='adam')
solver.fit(x_train1, epochs=200, batch_size=100)


# Generando datos para evaluación en el intervalo [-1, 1]
x_eval = np.linspace(-5, 5, 1000)
y_pred1 = solver.predict(x_eval)

# Trazar gráficas para la función b
plt.figure(figsize=(8, 6))
plt.plot(x_eval, y_pred1, color='red', label='Predictions')
plt.plot(x_eval, analytic_solu1(x_eval), color='green', linestyle='--', label='True function')
plt.title('Approximation de la fonction')
plt.legend()
plt.show()




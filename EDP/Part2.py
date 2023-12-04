import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout,Activation
from tensorflow.keras.optimizers import RMSprop, Adam

from matplotlib import pyplot as plt
import numpy as np


# Definicion de las funciones que tenemos que aproximar
def function_a (x) : 
    return 3 * np.sin(np.pi * x)

def function_b (x) : 
    return 1 + 2*x + 4*x**3

x_train = np.random.uniform(-1, 1, 1000)  # las valores aleatorias deben estar en el intervalo [-1, 1]
x_train = np.sort(x_train) # sort permite ordenanr las valores de 'x'
y_train_a = function_a(x_train)
y_train_b = function_b(x_train)

# Construccion del modélo con keras
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(1,)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')


# Para obtener el grafico de la funcion a, solo tienes que descommentar la parte siguiente
# y executar el condigo entero.
"""
# Entrenamiento del modélo para la funcion a
model.fit(x_train, y_train_a, epochs=100, verbose=0)

# Generando datos para evaluación en el intervalo [-1, 1]
x_eval = np.linspace(-1, 1, 1000)
y_pred_a = model.predict(x_eval)

# Trazar gráficas para la función a
plt.figure(figsize=(8, 6))
plt.scatter(x_train, y_train_a, label='Data')
plt.plot(x_eval, y_pred_a, color='red', label='Predictions')
plt.plot(x_eval, function_a(x_eval), color='green', linestyle='--', label='True function')
plt.title('Approximation de la fonction a')
plt.legend()
plt.show()

"""


# Para obtener el grafico de la funcion b, solo tienes que descommentar la parte siguiente
# y executar el condigo entero.
"""
# Entrenamiento del modélo para la funcion b
model.fit(x_train, y_train_b, epochs=100, verbose=0)

# Generando datos para evaluación en el intervalo [-1, 1]
x_eval = np.linspace(-1, 1, 1000)
y_pred_b = model.predict(x_eval)

# Trazar gráficas para la función b
plt.figure(figsize=(8, 6))
plt.scatter(x_train, y_train_b, label='Data')
plt.plot(x_eval, y_pred_b, color='red', label='Predictions')
plt.plot(x_eval, function_b(x_eval), color='green', linestyle='--', label='True function')
plt.title('Approximation de la fonction a')
plt.legend()
plt.show()
"""



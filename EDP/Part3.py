# Para obtener el grafico solo tiene que correr el codigo sigiente

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, Reshape
from tensorflow.keras.optimizers import RMSprop, Adam

from matplotlib import pyplot as plt
import numpy as np


class PolynomialLayer(Sequential):

    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.a0 = self.add_weight(shape=(1,), initializer='random_normal', trainable=True)
        self.a1 = self.add_weight(shape=(1,), initializer='random_normal', trainable=True)
        self.a2 = self.add_weight(shape=(1,), initializer='random_normal', trainable=True)
        self.a3 = self.add_weight(shape=(1,), initializer='random_normal', trainable=True)

    def call(self,x):
        return self.a0 + self.a1 * x + self.a2 * x**2 + self.a3 * x**3
    
# Definicion de las funciones que tenemos que aproximar
def function (x) : 
    return np.cos(2 * x)

# creacion de los datos de entrenamiento en el intervalo [-1,1]
x_train = np.random.uniform(-1, 1, 1000)
y_train = np.cos(2 * x_train)

# Creacion del modelo Sequential personalisado con PolynomialLayer 
model = tf.keras.Sequential([
    PolynomialLayer()
])

# compilacion del modélo
model.compile(optimizer='sgd', loss='mean_squared_error')

# entrenamiento del modelo con la funcion cos(2x)
model.fit(x_train, y_train, epochs=100, verbose=0)

# Generando datos para evaluación en el intervalo [-1, 1]
x_eval = np.linspace(-1, 1, 1000)
y_pred = model.predict(x_eval)

# Trazar gráficas para la función b
plt.figure(figsize=(8, 6))
plt.plot(x_eval, y_pred, color='red', label='Predictions')
plt.plot(x_eval, function(x_eval), color='green', linestyle='--', label='True function')
plt.title('Approximation de la fonction a')
plt.legend()
plt.show()


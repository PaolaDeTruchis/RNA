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

    def calcular(self,x):
        return self.a0 + self.a1 * x + self.a2 * x**2 + self.a3 * x**3
    

# creacion de los datos de entrenamiento en el intervalo [-1,1]
x_train = np.random.uniform(-1, 1, 1000)
y_train = np.cos(2 * x_train)

# Creacion del modelo con 
model = tf.keras.Sequential([
    PolynomialLayer()
])
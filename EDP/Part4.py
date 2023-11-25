import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, Reshape
from tensorflow.keras.optimizers import RMSprop, Adam

from matplotlib import pyplot as plt
import numpy as np

from PDESolver import PDESolver


def analytic_solu1 (x):
    return x**2 * np.sin(x) + x * np.cos(x)

x_train1 = np.random.uniform(-5, 5, 1000)  # las valores aleatorias deben estar en el intervalo [-5, 5]
x_train1 = np.sort(x_train1) # sort permite ordenanr las valores de 'x'
y_train1 = analytic_solu1(x_train1)


solver = PDESolver()

solver.compile(optimizer='adam')
solver.fit(x_train1, y_train1, epochs=10, batch_size=100)




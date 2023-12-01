import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, Reshape, Conv1D
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras import regularizers

from matplotlib import pyplot as plt
import numpy as np

from new_ODESolver import new_ODESolver
from PDESolver import PDESolver


def analytic_solu1 (x):
    return np.cos(x) - 0.5*np.sin(x)

x_train1 = np.random.uniform(0, 10, 1000)  
x_train1 = np.sort(x_train1) # sort permite ordenanr las valores de 'x'


solver = PDESolver()
solver.add(Dense(712, activation='tanh', kernel_regularizer=regularizers.l2(0.01))) 
solver.add(Dropout(0.2))
solver.add(Dense(356, activation='tanh', kernel_regularizer=regularizers.l2(0.01))) 
solver.add(Dropout(0.2))  # Añadi un layer  Dropout
solver.add(Dense(128, activation='tanh',kernel_regularizer=regularizers.l2(0.01)))
solver.add(Dense(1))

solver.build(input_shape=(None, 1))
solver.summary()

optimizer = Adam(learning_rate = 0.001)
solver.compile(optimizer=optimizer, loss='mse', metrics=['accuracy'])
solver.fit(x_train1, epochs=200, batch_size=100)


# Generando datos para evaluación en el intervalo [-1, 1]
x_eval = np.linspace(0, 10, 1000)
y_pred = solver.predict(x_eval)

# Trazar gráficas para la función b
plt.figure(figsize=(8, 6))
plt.plot(x_eval, y_pred, color='red', label='Predictions')
plt.plot(x_eval, analytic_solu1(x_eval), color='green', linestyle='--', label='True function')
plt.title('Approximation de la fonction a')
plt.legend()
plt.show()






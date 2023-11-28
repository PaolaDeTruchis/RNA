import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, Reshape
from tensorflow.keras.optimizers import RMSprop, Adam

from matplotlib import pyplot as plt
import numpy as np


# La clase PDESolver hereda de Sequential.
# Esta clase se utiliza para resolver ecuaciones en derivadas parciales.

class PDESolver(Sequential):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Inicializacion de las métricas de pérdida y el error cuadrático medio (MSE).
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.mse = tf.keras.losses.MeanSquaredError()

        self.add(Dense(64, activation='relu', input_shape=(2,)))
        self.add(Dense(128, activation='relu'))
        self.add(Dense(128, activation='relu'))
        self.add(Dense(1))

    # La propiedad 'metrics' devuelve la métrica de pérdida
    @property
    def metrics(self):
      return [self.loss_tracker]


    # Método de entrenamiento.
    def train_step(self, data):
        # determine el tamaño del batch
        batch_size=100

        # Generacion de datos aleatorios para 'x' y 't' dentro de ciertos rangos.
        x = tf.random.uniform((batch_size,1), minval=0, maxval=10)
        t = tf.random.uniform((batch_size,1), minval=0, maxval=15)

        # Determinacion de la pérdida y de las derivadas utilizando cintas de TensorFlow.
        with tf.GradientTape() as tape:
            # Cálculo de las derivadas en 'x' y 't'
            with tf.GradientTape(persistent=True) as g:
                g.watch(x)
                g.watch(t)

                with tf.GradientTape() as gg:
                    gg.watch(x)
                    input = tf.concat((x,t),axis=1)
                    y_pred = self(input, training=True)

                y_x=gg.gradient(y_pred,x)

            y_xx=g.gradient(y_x,x)
            y_t=g.gradient(y_pred,t)

            # Calculo de la ecuación en derivadas parciales (PDE).
            pde = x * y_x + y_pred - x * tf.cos(x)

            # Definicion de los valores iniciales y se calculo de la pérdida con el error cuadrático medio.
            t_init = tf.zeros(x.shape)
            input_ini = tf.concat((x,t_init),axis=1)
            y_init = self(input_ini, training=True)
            loss = self.mse(0., pde) + self.mse(tf.math.sin(x),y_init)

        # Compute grad
        grads = tape.gradient(loss, self.trainable_variables)

        # Update weights
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        # Compute our own metrics
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

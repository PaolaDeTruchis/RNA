import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Reshape
from tensorflow.keras.optimizers import RMSprop, Adam

from matplotlib import pyplot as plt
import numpy as np


# La clase PDESolver hereda de Sequential.
# Esta clase se utiliza para resolver ecuaciones en derivadas parciales.

class PDESolver(Sequential):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.mse = tf.keras.losses.MeanSquaredError()
        self.add(Dense(64, activation='relu', input_shape=(1,)))
        self.add(Dense(356, activation='relu'))     
        self.add(Dense(128, activation='relu'))
        self.add(Dense(1))

    @property
    def metrics(self):
      return [self.loss_tracker]

    def train_step(self, data):
        #batch_size=tf.shape(data)[0]
        batch_size=100

        x = tf.random.uniform((10,1), minval=0, maxval=10)


        with tf.GradientTape() as tape:
            #Loss value
            with tf.GradientTape(persistent=True) as g:
                g.watch(x)

                with tf.GradientTape() as gg:
                    gg.watch(x)
                    y_pred = self(x, training=True)

                y_x=gg.gradient(y_pred,x)
                #y_boundary = gg.gradient(y_pred,x0
            y_xx=g.gradient(y_x,x)

            pde =  y_xx + y_pred

            x_o = tf.zeros((batch_size,1))
            y_o = self(x_o,training=True)
            y_x_o = y_x
            y_init = 1
            y_x_init = -0.5
            loss = self.mse(tf.cast(0., tf.float32), pde) + self.mse(y_o , tf.cast(y_init, tf.float32)) + self.mse(y_x_o , tf.cast(y_x_init, tf.float32))


        # Compute grad
        grads = tape.gradient(loss, self.trainable_variables)

        # Update weights
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        # Compute our own metrics
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

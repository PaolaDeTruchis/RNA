import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Reshape, LSTM
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras import regularizers

from matplotlib import pyplot as plt
import numpy as np


# La clase PDESolver hereda de Sequential.
# Esta clase se utiliza para resolver ecuaciones en derivadas parciales.

class PDESolver(Sequential):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.mse = tf.keras.losses.MeanSquaredError()
        

    @property
    def metrics(self):
      return [self.loss_tracker]

    def train_step(self, data):
        #batch_size=tf.shape(data)[0]
        batch_size=100

        x = tf.random.uniform((batch_size, 1), minval=0, maxval=10)


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

            y_init = self(x, training=True)
            loss = self.mse(0., pde) + self.mse(tf.math.cos(x),y_init)


        # Compute grad
        grads = tape.gradient(loss, self.trainable_variables)

        # Update weights
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        # Compute our own metrics
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, Reshape
from tensorflow.keras.optimizers import RMSprop, Adam

from matplotlib import pyplot as plt
import numpy as np


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
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.mse = tf.keras.losses.MeanSquaredError()
        self.add(Dense(64, activation='relu', input_shape=(1,)))
        self.add(Dense(128, activation='relu'))
        self.add(Dense(1))

    @property
    def metrics(self):
      return [self.loss_tracker]

    def train_step(self, data):
        #batch_size=tf.shape(data)[0]
        batch_size=100

        x = tf.random.uniform((batch_size,1), minval=0, maxval=10)
        t = tf.random.uniform((batch_size,1), minval=0, maxval=15)


        with tf.GradientTape() as tape:
            #Loss value
            with tf.GradientTape(persistent=True) as g:
                g.watch(x)
                g.watch(t)

                with tf.GradientTape() as gg:
                    gg.watch(x)
                    input = tf.concat((x,t),axis=1)
                    y_pred = self(input, training=True)

                y_x=gg.gradient(y_pred,x)
                #y_boundary = gg.gradient(y_pred,x0)
            y_xx=g.gradient(y_x,x)
            y_t=g.gradient(y_pred,t)


            pde = y_t - 0.5*y_xx

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

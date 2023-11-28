import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout,Activation
from tensorflow.keras.optimizers import RMSprop, Adam

from matplotlib import pyplot as plt
import numpy as np

class new_ODESolver(Sequential):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Inicialización del seguimiento de la pérdida
        self.loss_tracker = keras.metrics.Mean(name="loss")
        # Uso del error cuadrático medio como función de pérdida
        self.mse = tf.keras.losses.MeanSquaredError()

        self.add(Dense(64, activation='relu', input_shape=(1,)))
        self.add(Dense(128, activation='relu'))
        self.add(Dense(1))

    # La linea siguiente permite definir un metedo que sea como un attribut 
    @property
    # Este metodo permite enviar una lista que contiene las 'metrics' de self.loss_tracker
    def metrics(self):
      return [self.loss_tracker]


    # Este funcion permite entrenar eln modelo
    # Genera datos aleatorios para el entrenamiento, gracias a la biblioteca numpy
    def train_step(self, data):
         
        # obtencion del tamaño del batch
        batch_size = tf.shape(data)[0]
        # Cálculo de los valores mínimos y máximos de los datos
        # para que generemos valores dentro del rango de los datos
        min = tf.cast(tf.reduce_min(data),tf.float32)
        max = tf.cast(tf.reduce_max(data),tf.float32)
        # Generamos valores aleatorios, gracias a numpy, a dentro del rango de los datos
        x = tf.random.uniform((batch_size,1), minval=min, maxval=max)

        # Calculo del gradiente, con el modelo, para resolver la EDO
        with tf.GradientTape() as tape:

            with tf.GradientTape(persistent=True) as g:
                g.watch(x)
                y_pred = self(x, training=True)# derivada del modelo con respecto a entradas x
            
            y_x = g.gradient(y_pred,x)
            x_o = tf.zeros((batch_size,1)) # valor de x en condicion inicial x_0=0
            y_o = self(x_o,training=True) # valor del modelo en en x_0
            eq = x * y_x + y_pred - x**2 * tf.cos(x) # Ecuacion diferencial evaluada en el modelo. Queremos que sea muy pequeno
            ic = 0.
            loss = self.mse(0., eq) + self.mse(y_o,ic)# calculo de la función de pérdida
 
        # Apply grads
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        # update metrics
        self.loss_tracker.update_state(loss)
        # Return a dict mapping metric names to current value
        return {"loss": self.loss_tracker.result()}



    
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout,Activation
from tensorflow.keras.optimizers import RMSprop, Adam


class ConvertToGrayscale(tf.keras.layers.Layer):
    def __init__(self):
        super(ConvertToGrayscale, self).__init__()

    def call(self, inputs):
        # Convertir la imagen a escala de grises
        grayscale = tf.image.rgb_to_grayscale(inputs)
        return grayscale
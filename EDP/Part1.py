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
    
# Ejemplo coon la base de datos MINST 

# Primero tenemos que cargar los dotas MINST
(train_images, train_labels), _ = tf.keras.datasets.mnist.load_data()

# Haora para que la red se antrena mas facilamente vamos a normalizar los datos.
# Para representar las fotos, cada pixel tiene un numero (entre 0 y 255) que craracterisa
# la nuancia de gris del mas obscuro (0 es negro) hasta el mas claro (255 es blanco).

train_images = train_images / 255.0   # Asi todos los valores son entre 0 y 1


# Ahora creo y entreno el modelo
model = keras.Sequential([ConvertToGrayscale(input_shape=(28, 28))])

model.summary()
model.compile(optimizer='adam')
model.fit(train_images, epochs=10, batch_size=100)








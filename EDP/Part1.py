import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential


class ConvertToGrayscale(keras.layers.Layer):
    def __init__(self):
        super(ConvertToGrayscale, self).__init__()

    def call(self, inputs):
        # Convertir la imagen a escala de grises
        grayscale = tf.image.rgb_to_grayscale(inputs)
        return grayscale
    
################# Ejemplo coon la base de datos MINST ######################## 

# Primero tenemos que cargar los dotas MINST
(train_images, train_labels), _ = tf.keras.datasets.mnist.load_data()

# Haora para que la red se antrena mas facilamente vamos a normalizar los datos.
# Para representar las fotos, cada pixel tiene un numero (entre 0 y 255) que craracterisa
# la nuancia de gris del mas obscuro (0 es negro) hasta el mas claro (255 es blanco).

train_images = train_images / 255.0   # Asi todos los valores son entre 0 y 1

# Tengo que cambiar la forma de los datos de entrenamiento.
# Primero me voy a visualizar el tamaño actual de esos datos :
print(train_images.shape) # la salida es : (60000, 28, 28)
# eso significa que tenemos 60000 imagenes de 28 por 28 pixeles

# Entonces tenemos que añadir un numero que especifica que el canal de color 
# en este ejemlo es 1 porque son nuancias de grises.
train_images = train_images.reshape(train_images.shape[0],train_images.shape[1],train_images.shape[2], 1)
print(train_images.shape)


# Creacion de un layer
convert_to_grayscale = ConvertToGrayscale()

# Ahora creo y entreno el modelo
model = Sequential([convert_to_grayscale])

model.summary()
model.compile(optimizer='adam')
model.fit(train_images, epochs=10, batch_size=100)








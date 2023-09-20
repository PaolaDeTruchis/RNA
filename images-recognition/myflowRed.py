#############################   IMPORT   #############################   

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.optimizers import RMSprop, SGD
from tensorflow.keras import regularizers


############################   SETTINGS   ############################ 

learning_rate = 0.001
epochs = 30
batch_size = 120


##############################   DATA   ##############################

"import of the training datas"
dataset=mnist.load_data()

(x_train, y_train), (x_test, y_test) = dataset

x_trainv = x_train.reshape(60000, 784)
x_testv = x_test.reshape(10000, 784)
x_trainv = x_trainv.astype('float32')
x_testv = x_testv.astype('float32')

x_trainv = x_trainv/255
x_testv = x_testv/255


#############################   NERWORK   #############################

num_classes=10
y_trainc = keras.utils.to_categorical(y_train, num_classes)
y_testc = keras.utils.to_categorical(y_test, num_classes)

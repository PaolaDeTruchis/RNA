#############################   IMPORT   #############################   

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.optimizers import RMSprop, SGD
from tensorflow.keras import regularizers


############################   SETTINGS   ############################ 

learning_rate = 0.01
epochs = 70
batch_size = 120
mini_batch_size = 15


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

num_classes=10  #number of possible output

#vectorization of responses
y_trainc = keras.utils.to_categorical(y_train, num_classes) 
y_testc = keras.utils.to_categorical(y_test, num_classes)


#############################   NERWORK   #############################


"""creation of dense sequential network"""
model = Sequential()        
model.add(Dense(50, activation='sigmoid', input_shape=(784,))) # creation of the first layer
model.add(Dense(num_classes, activation='sigmoid'))            # creation of the output layer

model.summary()     # visualization of the network


"""configuration of the model"""
model.compile(loss='categorical_crossentropy',optimizer=SGD(learning_rate=learning_rate),metrics=['accuracy'])  


"""training of the model"""
history = model.fit(x_trainv, y_trainc,
                    batch_size=mini_batch_size,
                    epochs=epochs,
                    verbose=1,            #show the result of each epochs
                    validation_data=(x_testv, y_testc))


"""evaluation of the model"""
score = model.evaluate(x_testv, y_testc, verbose=1) #evaluar la eficiencia del modelo
print(score) 




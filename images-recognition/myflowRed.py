#############################   IMPORT   #############################   

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Input
from tensorflow.keras.optimizers import RMSprop, SGD, Adam, Adadelta
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping


############################   SETTINGS   ############################ 

learning_rate = 0.01
epochs = 50
batch_size = 120
mini_batch_size = 10


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

"""Creacion of the callback EarlyStopping"""
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)


"""creation of dense sequential network"""
model = Sequential()        
model.add(Dense(196, activation='softplus', input_shape=(784,), kernel_regularizer=regularizers.L1(0.01)))
model.add(Dense(98, activation='softplus', kernel_regularizer=regularizers.L1(0.01)))
model.add(Dense(50, activation='softplus', kernel_regularizer=regularizers.L1(0.01)))
model.add(Dense(30, activation='softplus', kernel_regularizer=regularizers.L1(0.01)))
model.add(Dense(num_classes, activation='softmax', kernel_regularizer=regularizers.L1(0.01)))


"""Visualisation of the model"""
model.summary()     # visualization of the network


"""configuration of the model"""
model.compile(loss='categorical_crossentropy',optimizer=Adam(learning_rate=learning_rate),metrics=['accuracy'])  


"""training of the model, until it overadjust"""
history = model.fit(x_trainv, y_trainc,
                    batch_size=mini_batch_size,
                    epochs=epochs,
                    verbose=1,            #show the result of each epochs
                    validation_data=(x_testv, y_testc),
                    callbacks=[early_stopping])


"""conservation of the weights"""
model.save_weights("model_weights.h5")


"""creation of the new model"""
new_model = Sequential()        
new_model.add(Dense(196, activation='softplus', input_shape=(784,), kernel_regularizer=regularizers.L1(0.01)))
new_model.add(Dense(98, activation='softplus', kernel_regularizer=regularizers.L2(0.01)))
new_model.add(Dense(50, activation='softplus', kernel_regularizer=regularizers.L1L2(0.01,0.01)))
new_model.add(Dropout(0.3))
new_model.add(Dropout(0.3))
new_model.add(Dense(num_classes, activation='softmax', kernel_regularizer=regularizers.L1L2(0.01,0.01)))


"""training of the model, until it overadjust"""
history = new_model.fit(x_trainv, y_trainc,
                    batch_size=mini_batch_size,
                    epochs=epochs,
                    verbose=1,            #show the result of each epochs
                    validation_data=(x_testv, y_testc))


"""saving the weights in the new model"""
new_model.load_weights("model_weights.h5")


"""evaluation of the bestmodel"""
score = new_model.evaluate(x_testv, y_testc, verbose=1) #evaluar la eficiencia del modelo
print(score) 




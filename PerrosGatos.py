import os
import comet_ml
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Input,Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import RMSprop, SGD
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator


comet_ml.init(project_name="PerosyGatos")


experiment = comet_ml.Experiment(
    auto_histogram_weight_logging=True,
    auto_histogram_gradient_logging=True,
    auto_histogram_activation_logging=True,
    log_code=True,
)

parameters = {
    "batch_size": 128,
    "epochs": 15,
    "optimizer": "adam",
    "loss": "binary_crossentropy",
}
experiment.log_parameters(parameters)

batchsize = 128
img_height = 180
img_width = 180
epochs = 15
train_dir = r'train'
test_dir = r'test'
traindatagen = ImageDataGenerator (rescale=1./255, zoom_range=0.2, rotation_range=5, horizontal_flip=True)
train = traindatagen.flow_from_directory(train_dir, target_size=(img_width,img_height),batch_size=batchsize,class_mode='binary')
testdatagen = ImageDataGenerator(rescale=1./255)
test = testdatagen.flow_from_directory(test_dir, target_size=(img_width,img_height), batch_size=batchsize, class_mode='binary')




##################################### MODELO 1 #####################################
"""
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss=parameters['loss'], optimizer=parameters['optimizer'], metrics=['accuracy'])

model.compile(loss=parameters['loss'],optimizer=parameters['optimizer'],metrics=['accuracy'])
model.fit(train,
          batch_size=parameters['batch_size'],
          epochs=parameters["epochs"],
          verbose=1,
          validation_data=(test),
          callbacks=[checkpoint])
"""

##################################### MODELO 2 #####################################
# specify the path where you want to save the model
filepath = "/results2.h5"

# initialize the ModelCheckpoint callback
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

model2 = Sequential()
model2.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)))
model2.add(MaxPooling2D((2, 2)))
model2.add(Flatten())
model2.add(Dense(512, activation='relu'))
model2.add(Dropout(0.5))
model2.add(Dense(1, activation='sigmoid'))

model2.compile(loss=parameters['loss'], optimizer=parameters['optimizer'], metrics=['accuracy'])

model2.compile(loss=parameters['loss'],optimizer=parameters['optimizer'],metrics=['accuracy'])
model2.fit(train,
          batch_size=parameters['batch_size'],
          epochs=parameters["epochs"],
          verbose=1,
          validation_data=(test),
          callbacks=[checkpoint])

from comet_ml import Experiment
exp = Experiment()
exp.log_model("model2", "/results2.h5")


model2.save('model2.h5')
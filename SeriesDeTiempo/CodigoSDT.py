
### Imports 

import os
import numpy as np
from matplotlib import pyplot as plt
from tensorflow import keras



### Imprimimos los encabezados y separamos los datos del encabezado

fname = os.path.join("jena_climate_2009_2016.csv")
with open(fname) as f:
   data = f.read()

lines = data.split("\n")
header = lines[0].split(",")
lines = lines[1:]
print(header)
print(len(lines))


### Separamos datos de temperatura y los demás

temperature = np.zeros((len(lines),))
raw_data = np.zeros((len(lines), len(header) - 1))
for i, line in enumerate(lines):
  values = [float(x) for x in line.split(",")[1:]]
  temperature[i] = values[1]
  raw_data[i, :] = values[:]


### Graficamos la serie de tiempo de temperaturas

plt.plot(range(len(temperature)), temperature)


### Los datos son tomados cada 10 minutos. Tienen mucha resolución!!. 
### Por ejemplo, podemos graficar los primeros diez días. 24x6=144 <- Es 
### el número de datos correspondiente a un día.

plt.plot(range(1440), temperature[:1440])


### Analizar el siguiente código para identificar la funcionalidad de 
### la función : keras.utils.timeseries_dataset_form_array()

""""
int_sequence = np.arange(10) #Generamos un array de enteros del 0 al 9
dataset = keras.utils.timeseries_dataset_from_array(
    data=int_sequence[:-3],    #secuencia para valores de x
    targets = int_sequence[3:], #secuencia para extraer los valores y
    sequence_length=3,  #tamano de las secuencias x
    batch_size=2,  #cada vez que se llame "dataset" nos regresara un batch de 2 secuencias
)

for inputs, targets in dataset:
  print("series x")
  print(inputs)
  print("y")
  print(targets)
"""

### O bien, si queremos visualizar de forma mas ordenada
"""
for inputs, targets in dataset:
   for i in range(inputs.shape[0]):
     print([int(x) for x in inputs[i]], int(targets[i]))
"""


#########################################   TAREA   #########################################

### 1.   Genera secuencias x para entrenamiento (50%), validación (25%) y prueba (25%) de 120 
### elementos de longitud a partir de los datos de temperatura. 
### Aquí hay un ejemplo de una forma de lograrlo para generar un conjunto de secuencias:


val_dataset = keras.utils.timeseries_dataset_from_array(
    data=temperature[:-delay],
    targets=temperature[delay:],
    sampling_rate=sampling_rate,
    sequence_length=sequence_length,
    shuffle=True,
    batch_size=batch_size,
    start_index=num_train_samples,
    end_index=num_train_samples + num_val_samples)


print("c'est fini")


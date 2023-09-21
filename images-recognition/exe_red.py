import mnist_loader
import network 
import matplotlib.pyplot as plt
import numpy as np
import os

"""Usamos nuestra red, para reconocer las cifras a partir de fotos. 
Cada foto se compone de 784, por lo tanto, nuestro primero capa está 
componada de 784 neuronas. En la salida queremos saber cuál cifra 
está en la foto, en consecuencia necesitamos 10 salidas: 0,1,...,9.
No tenemos informaciones sobre las capas ocultas, entonces podemos 
elegir el número de capas ocultas y el número de neuronas en cada 
de esas capa. Aqui con la lista [784, 30, 10], hay solo una capa 
ocultada, que tiene 30 neuronas."""

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
net = network.Network([784, 30, 10]) # creacion de una red con tres capas 
                                     # previamente explicado


# entrenamiento de la red con los "training_data" de MNIST y pruebas con datos de pruebas
perf = net.SGD(list(training_data), 30, 10, 0.5, test_data=list(test_data)) 

perf = np.array(perf) # transformación de la lista de "perf" en una matriz

perf = 10000 - perf # permite minimizar (por convenciones)

plt.plot(perf,"ob") # crea el gráfico de perf 
#plt.savefig(          # guarda el gráfico  
#    os.path.abspath("C:/Users/Merci Charles/Documents/TEST/test_cost") + '/cost_quadratic error.png')
plt.show()            # mostra el gráfico




exit()
a=aplana(Imagen)
resultado = net.fedforward(a)
print (resultado)




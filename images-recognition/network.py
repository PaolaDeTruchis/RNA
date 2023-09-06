"""
network.py
~~~~~~~~~~

A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network.  Gradients are calculated
using backpropagation.  Note that I have focused on making the code
simple, easily readable, and easily modifiable.  It is not optimized,
and omits many desirable features.
"""

#### Libraries
# Standard library
"""Al principio, la red no está entrenada, por lo tanto, vamos a necesitar 
valores aleatorios para los w (weights) y los b (biases). Para obtener esos 
valores aleatorios usamos la librería random"""
import random


# Third-party libraries
"""Las redes neuronales artificiales se implementan con matrices. Por lo 
tanto, para implementar el código vamos a necesitar la librería numpy"""
import numpy as np



class Network(object):



    def __init__(self, sizes):
        """Esta función permite inicializar nuestra red. Como parámetros, damos 
        una lista, en la cual, cada número corresponde al número de neuronas en 
        la capa. Por ejemplo, size = [5, 10, 7, 3] es una red que tiene 5 neuronas 
        en su capa entrada, 3 en su capa de salida. Además, tiene 2 capas ocultadas, 
        con 10 y 7 neuronas"""    
        self.num_layers = len(sizes) # usamos 'len' para obtener el número 
                                     # de elementos en la lista en parámetro. 
                                     # Eso corresponde al número de capas
        self.sizes = sizes # esta variable coresponde a la lista que esta 
                           # en parametro y que contiene el numero de neuronas 
                           # por cada capa
        """La funcion 'np.random.randn(a,b)' permite crear y llenar una matrice, 
        de talla a x b, con nombres aleatorios entre 0 y 1.
        Usemos este funcion para crear matricies con valores aleatories. Hacemos 
        esto, porque al principio la red no esta entrenada, no sabemos nada de las 
        valores de biases y de weigths (solo que deben ser entre 0 y 1), por lo 
        tanto ponemos valores aleatorias entre 0 y 1"""
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]] 
                # este variable es una lista que se compone de vectores de talla 
                # de los números de la lista, a partir del segundo elemento. Esos 
                # vectores corresponden a los biases de cada capa, excepto la 
                # primera, porque es la entrada, entonces no tiene biases.          
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]
                # este variable es una lista que se compone de matrices de talla 
                # (talla de capa actual x talla de la proxima capa), a partir del 
                # segundo elemento. Esos matricies corresponden a los weigths de 
                # cada capa, excepto la ultima, porque es la salida, entonces no 
                # tiene weigths.


    def feedforward(self, a):
        """Esta función permite de recorrer la red capa por capa. 
        Los domas dos parámetros:
        1. 'self' que contiene las valores de weights y biases
        2. 'a' un vector que contiene los valores que entran en 
            este capa, es decir, la salida de las capas anteriores
        devuelve el valor de activación 'a', es decir, el valor obtenido 
        después del paso en la capa"""
        for b, w in zip(self.biases, self.weights): 
                # Gracias a este bucle 'for' recorremos cada capa (con el b 
                # en self.biaises) y cada neurona (con el w en self.weights). Por 
                # cada neurona obtenemos b : su bias (valor escalar) y w : los 
                # weigths de las enlaces que entran en la neurona (vector)
            z = np.dot(w, a)+b
                    # Calculemos 'z' : el valor de entrada gracias a 'np.dot' que 
                    # permite multiplicar matrices. Entonces hacemos el producto entre 
                    # 'a' (la salida de las neuronas anterior) y 'w' (los weigths) y 
                    # después añadimos 'b' (el bias)"""
            a = sigmoid(z)
                    #Al final calculemos la salida de la capa con la función sigmoid 
                    #que veremos después"""
        return a


    """Para entrenar nuestra red neuronal, utilizamos la función SGD (Stochastic 
    Gradient Descent). Además, para que el aprendizaje sea más rápido y el SGD 
    más estable, utilizamos un "mini-batch". Es decir, separaremos nuestros datos
      de entrenamiento en pequeños conjuntos de datos. Esto nos permitirá actualizar 
      nuestros sesgos y ponderaciones con más frecuencia."""
    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        """training_data :    corresponde a los datos de entrenamiente Se encuentran 
                              en forma de tupla (x,y) donde x son los datos de entrada 
                              en la red e y la salida esperada.
            epochs :          corresponde a cuántas veces el algoritmo de entrenamiento 
                              recorrerá el conjunto de datos de entrenamiento completo 
                              para entrenar la red.       
            mini_batch_size : es el tamaño de los pequeños conjuntos de datos
            eta :             (tasa de aprendizaje) este es el tamaño de paso utilizado 
                              para actualizar 'w' y 'b'
            test_data:        (datos de prueba, opcional) es un conjunto de datos que 
                              permite evaluar el rendimiento del modelo con datos 
                              diferentes a los utilizados para entrenar la red."""
        if test_data: n_test = len(test_data) # si tenemos datos de prueba, 
                                              # 'n_test' toma el tamaño de este 
                                              # conjunto de datos
        n = len(training_data) # n toma el valor del tamaño de los datos de entrenamiento
        for j in range(epochs): # este bucle permite recorrer el número de epocas
                                # entonces, todas las siguientes líneas de comando
                                # son recorriendo para cada epoca.
            random.shuffle(training_data) # esta línea permite mezclar los datos de 
                                          # entrenamiento para asegurarnos que sean 
                                          # repartidos en grupos aleatorios 
            mini_batches = [                            # este línea permite dividir los 
                training_data[k:k+mini_batch_size]      # datos de entrenamiento en grupos :
                for k in range(0, n, mini_batch_size)]  # "mini_batch" con un cierto tamaño 
                                                        # elegido : "mini_batch_size"
            for mini_batch in mini_batches:            # para cada "mini_batch" actualizamos 
                self.update_mini_batch(mini_batch, eta)# las valores de 'w' y 'b'
            if test_data:                                   # esas línea de codigo sirve para
                print ("Epoch {0}: {1} / {2}".format(       # generar 
                    j, self.evaluate(test_data), n_test))   # por cada epoca
            else:
                print ("Epoch {0} complete".format(j))
                

    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]


    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)

#### Miscellaneous functions
def sigmoid(z):
    return 1.0/(1.0+np.exp(-z)) #la funcion sigmoid

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z)) # derivada de la funcion sigmoid
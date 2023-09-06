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


    """Esta función permite inicializar nuestra red. Como parámetros, damos 
    una lista, en la cual, cada número corresponde al número de neuronas en 
    la capa. Por ejemplo, size = [5, 10, 7, 3] es una red que tiene 5 neuronas 
    en su capa entrada, 3 en su capa de salida. Además, tiene 2 capas ocultas, 
    con 10 y 7 neuronas""" 
    def __init__(self, sizes):   
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


    """Esta función permite de recorrer la red capa por capa. 
    Los domas dos parámetros:
    1. 'self' que contiene las valores de weights y biases
    2. 'a' un vector que contiene los valores que entran en 
        este capa, es decir, la salida de las capas anteriores
    devuelve el valor de activación 'a', es decir, el valor obtenido 
    después del paso en la capa"""
    def feedforward(self, a):
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
            for mini_batch in mini_batches:            # para cada "mini_batch" actualizamos las valores
                self.update_mini_batch(mini_batch, eta)# de 'w' y 'b' gracias a la funciona update_mini_batch
            if test_data:                                   # esas línea de codigo sirve para generar
                print ("Epoch {0}: {1} / {2}".format(       # el numero de buenas respuestas / el numero 
                    j, self.evaluate(test_data), n_test))   # total de respuestas por cada epoca, si hay 
            else:                                           # datos de prueba, de lo contrario, solo  
                print ("Epoch {0} complete".format(j))      # muestra el número epoca actual
                
    """Esta función: 'update_mini_batch' permite calcular los gradientes (gracias a la función backprop)
    y después con esos valores podemos calcular las nuevas valores de weigths y biases. Las fórmulas usadas 
    son fórmulas que hemos visto en clase"""
    def update_mini_batch(self, mini_batch, eta):
        """Esta función permite actualizar los 'w' y 'b', usando SGD y backprop"""
        nabla_b = [np.zeros(b.shape) for b in self.biases]   # nabla_b es una lista de vectores que 
                                                             # contienen los gradientes de los biases 
        nabla_w = [np.zeros(w.shape) for w in self.weights]  # nabla_w es una lista de matrices que 
                                                             # contienen los gradientes de los weigths
                                                             # para empezar llenemos ambos con ceros
                                                             # porque al principio no tenemos las valores   
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y) # calcule los deltas gracias a la función de backprop
            """La función 'zip' permite asociar elementos de dos listas. Este permite calcular los gradientes de 'b'
            y 'w' como lo hemos visto en clase"""
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)] 
        self.weights = [w-(eta/len(mini_batch))*nw                  # gracias a los gradientes de 'w' y a los weigths que ya 
                        for w, nw in zip(self.weights, nabla_w)]    # tenemos, podemos calcular las nuevas valores de weigths
        self.biases = [b-(eta/len(mini_batch))*nb                   # calculemos también el nuevo biases
                       for b, nb in zip(self.biases, nabla_b)]


    """Este funcion permite recorrer la red al revés para calcular los gradientes des los biases y de los weigths"""
    def backprop(self, x, y):
        """inicilizacion a cero de los gradientes 'b' y 'w'"""
        nabla_b = [np.zeros(b.shape) for b in self.biases]     
        nabla_w = [np.zeros(w.shape) for w in self.weights] 
        """feedforward (corremos la red para calcular y guardar en memoria las 
        activaciones de todas las neuronas)"""
        activation = x    # inicializacion con los valores de entrada 
        activations = [x] # creacion de una lista para despues poder añadir 
                          # facilamente los valores de activacion en la bucla 'for' 
        zs = [] # creacion de una lista para almanecer los valores de 'z', es decir
                # los valores de las entradas en cada capa
        for b, w in zip(self.biases, self.weights): # corremos todas las valores de 'b' y 'w'
                                                    # gracias a la funcion 'zip'
            z = np.dot(w, activation)+b  # calcul de 'z', el vector de las entradas 
            zs.append(z)                 # almacenamiento de 'z' en la lista 'zs'
            activation = sigmoid(z)         # calcul del valor de activacion de cada neurona
            activations.append(activation)  # almacenamiento de este valor en la lista 'activations'
        """backward pass (corremos la red a partir de la salida hasta la entrada para calcular 
        los gradientes.)"""
        #cacul del gradiente por la ultima capa 
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta                                         
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        #cacul del gradiente por las capas ocultadas
        for l in range(2, self.num_layers): # este 'for' recorre las capas ocultadas al revés
            z = zs[-l]  # recupera 'z', los valores de activacion 
            sp = sigmoid_prime(z) # calcule 'sp' la derivada de la activacion  
            # cálculo del nuevo gradiente 'delta' usando el gradiente de la siguiente capa multiplicado 
            # por los pesos de la capa actual, todo multiplicado por la derivada de la función de activación.
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp  
            nabla_b[-l] = delta # calcul del nuevo gradiente de 'b'
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose()) # calcul del nuevo gradiente de 'w'
        """devuelve el tuple de gradientes de los biases y de los weigths"""
        return (nabla_b, nabla_w)
    

    """Esta función devuelve el número de entradas, de los datos de prueba para 
    las cuales, la red produce el resultado correcto."""
    def evaluate(self, test_data):
        # test_results es una lista que contiene los tuples (x,y) donde 'x' es la salida 
        # obtenida con la red, con los datos de pruebas y 'y' esta es la verdadera respuesta
        test_results = [(np.argmax(self.feedforward(x)), y) # esta línea permite calcular la cifra 
                                                            # que tiene la más grande probabilidad 
                                                            # a la salida de la red, es decir, la 
                                                            # cifra reconocida por la red
                        for (x, y) in test_data] # este bucle 'for' cumple la línea arriba
                                                 # por cada tuple (x, y) en los datos de prueba
        return sum(int(x == y) for (x, y) in test_results) # Calcule el número de respuesta correcta 

    def cost_derivative(self, output_activations, y):
        """Devuelve el vector de derivadas parciales \partial C_x /
        \partial a para las activaciones de salida."""
        return (output_activations-y)
    

#### Miscellaneous functions
def sigmoid(z):
    return 1.0/(1.0+np.exp(-z)) # funcion sigmoid, usada para calcular 
                                # las activaciones de las neuronas

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z)) # derivada de la funcion sigmoid, usada 
                                     # para calcular los gradientes

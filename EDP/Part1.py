import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout,Activation
from tensorflow.keras.optimizers import RMSprop, Adam

from matplotlib import pyplot as plt
import numpy as np

from ODEsolver import ODEsolver

# Definicion de las funciones que tenemos que aproximar
def function_a (x) : 
    return 3 * np.sin(np.pi * x)

def function_b (x) : 
    return 1 + 2*x + 4*x**3



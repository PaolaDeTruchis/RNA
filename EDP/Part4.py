import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, Reshape
from tensorflow.keras.optimizers import RMSprop, Adam

from matplotlib import pyplot as plt
import numpy as np


# Ecuación diferencial 1 (isolando y')
def f1(x, y):
    return (x**2 * np.cos(x) - y) / x

# Condiciones iniciales 
x_0 = 0
y_0 = 0
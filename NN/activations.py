from activation import activation
import numpy as np

class Tanh(activation):
    def __init__(self):
        def tanh(x):
            return np.tanh(x)
        def tanh_prime(x):
            return 1 - np.tanh(x) ** 2
        super().__init__(tanh, tanh_prime)


class  Sigmoid(activation):
    def __init__(self):
        def sigmoid(x):
            return 1/(1+np.exp(-x))
        def sigmoid_prime(x):
            s = sigmoid(x)
            return s * (1 - s)
        super().__init__(sigmoid, sigmoid_prime)


        
from layer import Layer
import numpy as np

class activation(Layer):
    def __init__(self, activation, activation_prime):
        # here activaitin_prime denotes the derivative of the activation
        self.activation = activation
        self.activation_prime = activation_prime
    
    def forward(self, input):
        self.input = input
        return self.activation(self.input)
    
    def backward(self, output_gradient,learning_rate):
        return np.multiply(output_gradient, self.activation_prime(self.input))
